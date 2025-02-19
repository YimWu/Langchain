from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os

# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建ChatOpenAI实例，指定模型名称、API Key和base_url
chat = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 定义系统消息和用户消息
messages = [
    HumanMessage(content="请给我写一句情人节红玫瑰的中文宣传语")
]

# 使用invoke方法来获取回复
response = chat.invoke(messages)

# 打印响应内容
print(response.content)