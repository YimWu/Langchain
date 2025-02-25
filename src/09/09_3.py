from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from langchain.schema.runnable import RunnableLambda

# 设置 API 密钥和基础 URL
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建 ChatOpenAI 实例
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.2
)

# 创建 PromptTemplate
template1 = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个 200 字左右的介绍。

花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:"""

template2 = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇 200 字左右的评论。

鲜花介绍:
{introduction}
花评人对上述花的评论:"""

template3 = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300 字左右。

鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}

社交媒体帖子:
"""

prompt_template1 = PromptTemplate(input_variables=["name", "color"], template=template1)
prompt_template2 = PromptTemplate(input_variables=["introduction"], template=template2)
prompt_template3 = PromptTemplate(input_variables=["introduction", "review"], template=template3)

# 定义子链
chain1 = prompt_template1 | llm
chain2 = prompt_template2 | llm
chain3 = prompt_template3 | llm

# 将 lambda 函数转换为 RunnableLambda
lambda_step = RunnableLambda(lambda x: {"introduction": x["introduction"], "review": x["review"]})

# 构建主链
overall_chain = (
    {
        "introduction": chain1,
        "review": chain1 | chain2
    }
    | lambda_step
    | chain3
)

# 运行链，并打印结果
res = overall_chain.invoke({"name": "玫瑰", "color": "黑色"})

print(res.content)