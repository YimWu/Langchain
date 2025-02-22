# 导入所需要的库和模块
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import OutputFixingParser
import os
from langchain_openai import ChatOpenAI

# 1.使用Pydantic创建一个数据格式，表示花
class Flower(BaseModel):
    name: str = Field(description="name of a flower")
    colors: List[str] = Field(description="the colors of this flower")
# 2.定义一个用于获取某种花的颜色列表的查询
flower_query = "Generate the charaters for a random flower."

# 3.定义一个格式不正确的输出,单引号应该改为双引号
misformatted = "{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"

# 4.创建一个用于解析输出的Pydantic解析器，此处希望解析为Flower格式
parser = PydanticOutputParser(pydantic_object=Flower)
# # 使用Pydantic解析器解析不正确的输出,报错
# parser.parse(misformatted)


# 5.使用OutputFixingParser创建一个新的解析器，该解析器能够纠正格式不正确的输出
# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 创建ChatOpenAI实例，指定模型名称、API Key和base_url
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)

new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

# 使用新的解析器解析不正确的输出
result = new_parser.parse(misformatted) # 错误被自动修正
print(result) # 打印解析后的输出结果
