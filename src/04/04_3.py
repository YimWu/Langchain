import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd

# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 1. 创建提示模板（基础提示词+变量+输出指示变量）
prompt_template = """您是一位非常专业的鲜花销售文案撰写大师。
对于售价{price}元的{flower_name},给出一个吸引人的营销文案
{format_instructions}
"""
# 2. 定义响应模式
response_schemas = [
    ResponseSchema(name="description", description="描述文案"),
    ResponseSchema(name="reson",description="为什么这么写")
]
# 3. 创建输出解释器（传入响应模式）
output_parse = StructuredOutputParser.from_response_schemas(response_schemas)
# 4. 通过输出解释器获取输出指示
format_instructions = output_parse.get_format_instructions()
# 5. 创建提示词模板实例（传入提示词模板+输出指示）
prompt = PromptTemplate.from_template(prompt_template,
                                      partial_variables={"format_instructions": format_instructions})
# 6. 提示词模板实例（传入变量），获取提示词实例
flower = "玫瑰"
price = "999"
input = prompt.format(flower_name=flower,price=price)
# 7. 调用大模型（传入提示词实例）
model = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)
# 8. 调用输出解释器解析大模型输出
output = model.invoke(input).content
parsed_output = output_parse.parse(output)
print(parsed_output)