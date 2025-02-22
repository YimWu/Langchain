from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from openai import OpenAI, APIError
import os
import logging
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

# 1.加载Documents
base_dir = './assets' # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
print("1-All data has been loaded!")

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
print("2-All data has been splited!")

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
# 创建 AliyunEmbeddings 用于替代 OpenAiEmbeddings
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

vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=AliyunEmbeddings(), # 用AliyunEmbeddings Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name
print("3.Save chunked_documents to vectorstore!")
# 4.准备模型
# 创建ChatOpenAI实例，指定模型名称、API Key和base_url
# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)
print("4.Created LLM!")
# 5.实例化一个MultiQueryRetriever
# 设置Logging,不设置则不展示
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
print("5.Created MultiQueryRetriever!")
# 6.实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)
print("6.Created RetrievalQA链!")
# 7.调用print("4.Created LLM!") RetrievalQA 读入问题，生成答案
question = "董事长强调的公司愿景是什么"
response = qa_chain.invoke({"query": question})
print("7.Get response!")
print(response)