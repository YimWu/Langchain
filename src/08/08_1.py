from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd
import os
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# 1.设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 2.创建ChatOpenAI实例，指定模型名称、API Key和base_url
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 3.创建一个空的DataFrame用于存储结果
df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = [50, 30, 20]

# 4.定义我们想要接收的数据格式
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")
# 创建输出解析器
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)
# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
# print("输出格式：",format_instructions)

# 5.创建提示模板
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template,
       partial_variables={"format_instructions": format_instructions})

# 打印提示
# print("提示：", prompt)

# 6.生成结果并输出
for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower=flower, price=price)
    # 打印提示
    # print("提示：", input)
    # 获取模型的输出
    output = llm.invoke(input).content
    # 解析模型的输出
    parsed_output = output_parser.parse(output)
    parsed_output_dict = parsed_output.model_dump()  # 将Pydantic格式转换为字典
    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output.model_dump()
# 打印字典
print("输出的数据：", df.to_dict(orient='records'))