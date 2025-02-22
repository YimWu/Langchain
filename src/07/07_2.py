from transformers import AutoTokenizer, pipeline
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# 指定预训练模型的名称
model = "meta-llama/Meta-Llama-3-8B"

# 从预训练模型中加载词汇器，并设置截断策略
tokenizer = AutoTokenizer.from_pretrained(model)

# 创建一个文本生成的管道
pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1000,
)

# 使用更新后的类创建实例
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

# 定义输入模板
template = """
              为以下的花束生成一个详细且吸引人的描述：
              花束的详细信息：
              ```{flower_details}```
           """

# 使用模板创建提示
prompt = PromptTemplate(template=template, input_variables=["flower_details"])

# 创建LLMChain实例
llm_chain = prompt | llm

# 需要生成描述的花束的详细信息
flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"

# 打印生成的花束描述
print(llm_chain.invoke({"flower_details": flower_details}))