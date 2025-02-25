from langchain_openai import ChatOpenAI
import os
from langchain.prompts import PromptTemplate
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.2
)
template = "{flower}的花语是什么"
prompt_temp = PromptTemplate.from_template(template)
prompt = prompt_temp.format(flower="玫瑰")
print("prompt", prompt)
res = llm.invoke(prompt)
print(res.content)