from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from openai import OpenAI, APIError
from langchain.embeddings.base import Embeddings  # 确保正确导入Embeddings接口
import os
# 1. 创建示例样本
samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]
# 2. 创建提示模板
template = "鲜花类型：{flower_type}\n场合：{occasion}\n文案：{ad_copy}"
prompt_sample = PromptTemplate(input_variables=["flower_type", "occasion", "ad_copy"],
                               template=template)
print("Sample")
print(prompt_sample.format(**samples[0]))

# 3.初始化示例选择器
# 创建AliyunEmbeddings类
class AliyunEmbeddings(Embeddings):
    def __init__(self, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _get_embedding(self, text):
        """获取单个文本的嵌入向量"""
        try:
            completion = self.client.embeddings.create(
                model="text-embedding-v3",
                input=text,
                dimensions=1024,
                encoding_format="float"
            )
            response_data = completion.model_dump()
            if 'data' not in response_data or len(response_data['data']) == 0 or 'embedding' not in response_data['data'][0]:
                print(f"Unexpected response format for text '{text[:50]}...': {response_data}")
                return None
            return response_data['data'][0]['embedding']
        except APIError as e:
            print(f"API call failed for text '{text[:50]}...' with error: {e}")
            return None

    def embed_query(self, text):
        """嵌入单个查询文本"""
        embedding = self._get_embedding(text)
        if embedding is None:
            raise ValueError(f"Failed to get embedding for query text: {text}")
        return embedding

    def embed_documents(self, texts):
        """嵌入一系列文档"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                print(f"Failed to get embedding for document text: {text[:50]}...")
        return embeddings

    def __call__(self, texts):
        """使AliyunEmbeddings实例可以像函数一样被调用"""
        if isinstance(texts, str):
            # 如果传入的是单个字符串，则视为单个文本
            return self.embed_query(texts)
        elif isinstance(texts, list):
            # 如果传入的是列表，则视为多个文本
            return self.embed_documents(texts)
        else:
            raise ValueError("Input must be a string or a list")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    AliyunEmbeddings(),
    Chroma,
    k=1
)
# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"]
)
print("\nSamples")
print(prompt.format(flower_type="野玫瑰",occasion="爱情"))
# 4. 调用大模型创建新文案
# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)
result = model.invoke(prompt.format(flower_type = "野玫瑰", occasion = "爱情"))
print("\nFewShot-Result")
print(result.content)
