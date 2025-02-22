from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser, RetryWithErrorOutputParser
from langchain_openai import ChatOpenAI
import os

# 1.使用Pydantic格式Action来初始化一个输出解析器
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")
parser = PydanticOutputParser(pydantic_object=Action)

# 2.定义一个提示模板，它将用于向模型提问
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="What are the colors of Orchid?")

# 3.定义一个错误格式的字符串
bad_response = '{"action": "search"}'
# parser.parse(bad_response) # 如果直接解析，它会引发一个错误

# 4.创建ChatOpenAI实例，指定模型名称、API Key和base_url
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0
)

# 5.使用OutputFixingParser创建一个新的解析器，该解析器能够纠正格式不正确的输出
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
parse_result = fix_parser.parse(bad_response)
print('OutputFixingParser的parse结果:',parse_result)

# 6.初始化RetryWithErrorOutputParser，它会尝试再次提问来得到一个正确的输出
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser, llm=llm
)
parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print('RetryWithErrorOutputParser的parse结果:',parse_result)

