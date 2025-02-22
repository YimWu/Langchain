from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# 初始化HF LLM
llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B", # 确保repo_id对应于一个有效的、支持'text2text-generation'任务的模型
)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

# 使用新的方式创建链
llm_chain = prompt | llm

# 准备问题
# question = "Rose is which type of flower?"
question = "请讲一个李伟强的故事"

# 调用模型并返回结果
response = llm_chain.invoke({"question": question})
print(response)