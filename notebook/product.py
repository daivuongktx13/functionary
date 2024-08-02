import os
import json
from random import sample

os.environ['OPENAI_API_KEY'] = "xx"

from typing import Annotated, List, Optional
import pandas as pd

from langchain.schema import Document
from langchain_core.tools import tool, ToolException
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from sql.product import df_products, find_db_products_by_properties

def get_metadata(product: dict):
    metadata = dict()
    fields = ['product_name', 'product_id', 'frst_level_cat', 'scnd_level_cat', 'thrd_level_cat']
    
    for key in fields:
        metadata[key] = product[key]

    return metadata

products = []
for i, row in df_products.iterrows():
    products.append(row.to_dict())

shop_product_ids = set( [str(x) for x in df_products['product_id'].tolist()] )
product_id_to_details = dict(zip(df_products['product_id'].tolist(), df_products.to_dict('records')))

def format_product_content(product: dict):
    return f"""Sản phẩm: {product['product_name']}\nNgành hàng: {product['thrd_level_cat']}"""

product_docs = [Document(page_content=format_product_content(p), metadata=get_metadata(p)) for p in products]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
product_db = FAISS.from_documents(product_docs, embeddings)
product_retriever = product_db.as_retriever(search_kwargs={'k': 5})

@tool
def find_products_by_properties(
    product_code: Annotated[str, "Mã code của sản phẩm, ví dụ: A1, B2, C3, ..."] = None, 
    color: Annotated[str, "Màu của sản phẩm bằng Tiếng Việt utf-8: xanh, đỏ, tím, ..."] = None, 
    size: Annotated[str, "Kích cỡ sản phẩm: S, L, XL, ..."] = None
):
    """Tìm kiếm các sản phẩm dựa trên màu sắc hoặc kích cỡ."""
    found_products =  find_db_products_by_properties(product_code, color, size)
    found_products_dicts = [product.to_dict() for product in found_products]
    if len(found_products_dicts) == 0:
        content = f'Không có sản phẩm nào với product_code: {product_code}, color: {color}, size: {size}'
    else:
        content = json.dumps(found_products_dicts, ensure_ascii=False)
    return {
        'name': find_products_by_properties.name,
        'content': content
    }
@tool
def find_products_by_name(
    name: Annotated[str, "Tên sản phẩm thường theo format ví dụ [code màu size (kèm theo)]: Đ69 đỏ tươi M (kèm quần)"]
):
    """Tìm kiếm các sản phẩm dựa trên tên sản phẩm"""
    docs = product_retriever.invoke(name)
    found_products_dicts = [x.metadata for x in docs]
    if len(found_products_dicts) == 0:
        content = f'Không có sản phẩm nào với name: {name}'
    else:
        content = json.dumps(found_products_dicts, ensure_ascii=False)
    return {
        'name': find_products_by_properties.name,
        'content': content
    }

####
@tool
def get_random_products(k: int) -> list[dict]:
    """Lấy ra k sản phẩm ngẫu nhiên của shop."""
    random_products = sample(products, k)
    return {
        'name': get_random_products.name,
        'content': json.dumps([get_metadata(x) for x in random_products], ensure_ascii=False)
    }

@tool
def get_product_details(
    product_id: Annotated[str, "product_id của sản phẩm được lấy từ các hàm tìm kiếm products"]
):
    """Lấy thông tin chi tiết (tên, giá) của một sản phẩm theo id"""
    if product_id not in shop_product_ids:
        return {
            'name':  get_product_details.name,
            'content': f"Invalid product_id {product_id}"
        }
    
    product_name = product_id_to_details[int(product_id)]['product_name']
    price = (hash(product_id_to_details[int(product_id)]['product_preprocess'][0]['type']) % 40 + 50) * 10000 # random same price based on code (unit: VND)

    return {
        'name': get_product_details.name,
        'content': {
            "Tên sản phẩm": product_name, 
            "Giá": f"{price} VND",
        }
    }

@tool
def close_the_deal(
    address: Annotated[str, "Địa chỉ giao hàng"],
    phone_number: Annotated[str, "Số điện thoại giao hàng"],
    product_ids: Annotated[List[str], "List các product_id của các sản phẩm khách chốt mua."]
):
    "Chốt đơn hàng cho khách."
    for p_id in product_ids:
        if p_id not in shop_product_ids:
            return {
            'name':  close_the_deal.name,
            'content': f"Invalid product_id {p_id}"
        }
    content = "Chốt đơn thành công.\n"
    content += f"SĐT: {product_ids}\n"
    content += f"Địa chỉ: {address}\n"
    
    for p_id in product_ids:
        content += f"[{p_id}] {product_id_to_details[int(p_id)]['product_name']}\n"

    return {
            'name':  close_the_deal.name,
            'content': content
        }
