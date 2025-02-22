from langchain_openai import ChatOpenAI
import os
from langchain.prompts import PromptTemplate
# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 1.创建原始模板
template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 2.根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template)
# 打印LangChain提示模板的内容
# print(prompt)

# 3.创建ChatOpenAI实例，指定模型名称、API Key和base_url
chat = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 4.通过循环调用批量生成
flowers = ["玫瑰","百合","康乃馨"]
prices = ["50","30","20"]
for flower,price in zip(flowers, prices):
    # 输入提示
    input_prompt = prompt.format(flower_name=flower,price=price)
    # 使用invoke方法来获取回复
    response = chat.invoke(input_prompt)
    # 打印响应内容
    print(response.content)