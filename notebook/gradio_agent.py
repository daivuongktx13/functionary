# %%
import json
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

import gradio as gr

from product import get_random_products, find_products_by_name, find_products_by_properties, get_product_details, close_the_deal

# %%
llm = ChatOpenAI(model="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    base_url="http://localhost:8000/v1", api_key="functionary")

# %%
@tool
def calculate_shipping_fee(address: str) -> str:
    """Xác định phí ship dựa vào thông tin địa chỉ nhận hàng của khách."""
    return {
        'name': calculate_shipping_fee.name,
        'content': "Ship nội thành trong phạm vi 5km: free ship"
    }

@tool
def get_current_datetime() -> str:
    """Lấy thông tin thời gian hiện tại dưới dạng ISO Date Time Format."""
    return {
        'name': get_current_datetime.name,
        'content': datetime.now().isoformat()
    }

tools = [
    get_current_datetime, 
    get_random_products,
    find_products_by_properties, 
    find_products_by_name,
    get_product_details,
    close_the_deal]

# %%
prompt = [
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad')
]

prompt = ChatPromptTemplate.from_messages(prompt)

# %%
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parse_errors=True,
    return_intermediate_steps=True,
    debug=True)

def gradio_response(message, history, chat_history):
    # print(strftime("%Y-%m-%d %H:%M:%S"))
    user_query = message

    if len(user_query) == 0: return "Bấm gì đó đi chứ"
    ai_response = agent_executor.invoke(
        {
            "input": user_query,
            "chat_history": chat_history,
        }
    )
    chat_history.append(HumanMessage(content=user_query))

    # TEST
    wait_count = 0
    wait_total = 0
    for action, observation in ai_response['intermediate_steps']:
        tool_calls = action.message_log[0].tool_calls
        content = action.message_log[0].content
        if wait_count == 0: # handle for both parallel tool call + sequential single tool call
            wait_total = len(tool_calls)
            wait_count = len(tool_calls)
            chat_history.append(AIMessage(content=content, tool_calls=tool_calls))
        i = wait_total - wait_count
        wait_count -= 1
        chat_history.append(
            ToolMessage(content=json.dumps(observation, ensure_ascii=False), 
            tool_call_id=tool_calls[i]['id'])
        )
    ###
    
    chat_history.append(AIMessage(content=ai_response['output']))
    
    return ai_response['output']

gr.ChatInterface(fn=gradio_response, additional_inputs=[gr.State([])], theme='soft', title='Tiệm áo dài 24/7', retry_btn=None, undo_btn=None, clear_btn=None).launch(
    auth=("test", "Xxso8xIV_H@k0H9"),
    server_name="0.0.0.0", 
    server_port=3115,
)