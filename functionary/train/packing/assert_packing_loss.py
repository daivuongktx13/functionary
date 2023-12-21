import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import math
import typer
import sys
from packed_dataset import PackedDataset


random.seed(1)
torch.manual_seed(3)


def compute_loss_of_model(
    model: Any, ds: Dataset, tokenizer: Any, batch_size=8
) -> Tuple[float, int]:
    """Compute the avg loss per token given the model and dataset
    also return the number of tokens for computing loss

    Args:
        model (Any): model to compute the loss
        ds (Dataset): dataset to compute the loss
        tokenizer (Any): Tokenizer
        batch_size (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    total_loss = 0
    model.eval()

    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_num_loss_tokens = 0  # this is the total number of tokens for computing loss

    for index, batch in enumerate(data_loader):
        print(f"compute loss for batch: {index}")
        for key in batch:
            batch[key] = batch[key].to(model.device)

        if "labels" not in batch:
            labels = batch["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[labels == tokenizer.bos_token_id] = -100
            batch["labels"] = labels

        batch["return_dict"] = True

        with torch.no_grad():
            avg_loss = model.forward(**batch).loss.item()
            # compute number of tokens used for computing loss
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            ignore_count = (shift_labels == -100).sum()
            num_tokens = shift_labels.size(0) - ignore_count

            total_num_loss_tokens += num_tokens.item()
            total_loss += avg_loss * num_tokens.item()
    return total_loss / total_num_loss_tokens, total_num_loss_tokens


def compute_loss_for_model_class(
    pretrained_path: str, model_class: Any, tokenizer: Any, ds: Any
) -> Tuple[float, int]:
    """Compute the loss with model initilized from model_class
        also return the number of tokens for computing the loss
    Args:
        pretrained_path (str): model_path
        model_class (Any): model_class to initialize model, can be monkey-patched class or original class
        tokenizer (Any): _description_
        ds (Any): _description_

    Returns:
        _type_: _description_
    """
    model = model_class.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if hasattr(model, "router_aux_loss_coef"):
        print("set au_coef=0")
        model.router_aux_loss_coef = (
            0  # excluding auxilary loss (in MoE model) in comparison
        )

    return compute_loss_of_model(model, ds, tokenizer)


def create_labels_from_input_ids(input_ids: List[int], tokenizer: Any) -> List[int]:
    """Mask all token_ids to -100 except token_ids of output

    Args:
        input_ids (List[int]): input_ids
        tokenizer (Any): tokenizer

    Returns:
        _type_: _description_
    """
    labels = list(input_ids)
    output_prefix = tokenizer.encode("\n### Response:", add_special_tokens=False)
    # Sometimes Llamatokenizer adds 29871 (in Llama2-model) and 28705 (in Mistal-model), we need to remove
    if output_prefix[0] in [28705, 29871]:
        output_prefix = output_prefix[1:]
    index = None  # find the index of output_prefix
    for i in range(len(input_ids)):
        if input_ids[i : i + len(output_prefix)] == output_prefix:
            index = i + len(output_prefix)
            break
    # Mask input_ids until token_ids of: "\n### Response:"
    for i in range(index):
        labels[i] = -100
    return labels


def main(
    pretrained_path: str,
    max_input_length: int = typer.Option(default=4096),
    pack_length: int = typer.Option(default=4096),
    masking_labels: bool = typer.Option(default=False),
):
    """_summary_

    Args:
        pretrained_path (str): model_path
        max_input_length (int, optional): max_length at tokenizing data. Defaults to 4096.
        pack_length (int, optional): the length used for packing. Defaults to 6000.
        masking_labels: whether we mask labels such that only Output tokens are used for computing the loss
    Returns:
        _type_: _description_
    """

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer: ", tokenizer)

    model_config = transformers.AutoConfig.from_pretrained(pretrained_path)
    config_type = type(model_config).__name__.lower()
    if "mistral" in config_type:
        print("model: Mistral ")
        from mistral_monkey_patch import MistralForCausalLM

        mk_model_class = MistralForCausalLM
    elif "llama" in config_type:
        print("model: Llama ")
        from llama_monkey_patch import LlamaForCausalLM

        mk_model_class = LlamaForCausalLM
    elif "mixtral" in config_type:
        # Mixtral requires this ?
        tokenizer.padding_side = "left"
        print("model: Mixtral")
        from mixtral_monkey_patch import MixtralForCausalLM

        mk_model_class = MixtralForCausalLM
    else:
        print(
            f"{config_type} is not supported, currently we only support: Mistral, Mixtral, Llama"
        )
        sys.exit(1)

    ds = load_dataset("tatsu-lab/alpaca")["train"]
    # extract 100 random data points from ds
    size = len(ds)
    indices = [i for i in range(size)]
    random.shuffle(indices)

    # We randomly select 150 data points for computing loss
    ex_ds = ds.select(indices[:100])
    print("number of data points: ", len(ex_ds))
    original_columns = ex_ds.column_names

    def process_data(examples):
        prompts = examples["text"]
        input_dic = tokenizer(
            prompts, max_length=max_input_length, padding="max_length", truncation=True
        )

        if masking_labels:
            # create labels by masking from start to: "### Response:"
            batch_input_ids = input_dic["input_ids"]
            batch_labels = []
            for input_ids in batch_input_ids:
                labels = create_labels_from_input_ids(input_ids, tokenizer)
                batch_labels.append(labels)
            input_dic["labels"] = batch_labels

        return input_dic

    ex_ds = ex_ds.map(process_data, batched=True, remove_columns=original_columns)
    ex_ds.set_format("torch")

    packed_ds = PackedDataset(ex_ds, tokenizer, pack_length)
    packed_ds.stat()

    mk_avg_loss, mk_token_count = compute_loss_for_model_class(
        pretrained_path, mk_model_class, tokenizer, packed_ds
    )
    print("monkey_patched loss: ", mk_avg_loss)
    original_model_class = transformers.AutoModelForCausalLM

    original_avg_loss, original_token_count = compute_loss_for_model_class(
        pretrained_path, original_model_class, tokenizer, ex_ds
    )
    print("original_loss: ", original_avg_loss)

    # Make sure that number of tokens used for computing loss are the same in original dataset and packed dataset
    assert (
        original_token_count == mk_token_count
    ), f"number of tokens for computing loss is different: original_token_count = {original_token_count}, mk_token_count={mk_token_count}"
    diff_loss = math.fabs(mk_avg_loss - original_avg_loss) / original_avg_loss
    print(
        f"original_loss: {original_avg_loss}, monkey-patched loss: {mk_avg_loss}, diff={diff_loss:2.4f}%"
    )


if __name__ == "__main__":
    typer.run(main)