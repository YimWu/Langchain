from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
# 设置API密钥和基础URL
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建ChatOpenAI实例
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.2
)
# 创建PromptTemplate
template = "{flower}的花语是什么"
prompt = PromptTemplate(template=template, input_variables=["flower"])
# 使用LLMChain创建链，并直接调用
chain = prompt | llm
# 执行链并获取结果
response = chain.invoke({"flower": "玫瑰花"})
print(response.content)