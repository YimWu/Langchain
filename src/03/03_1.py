import os
# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings  # 确保正确导入Embeddings接口
from openai import OpenAI, APIError
import os
import logging # 导入Logging工具
from langchain_openai import ChatOpenAI # ChatOpenAI模型
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链
from flask import Flask, request, render_template

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
vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=AliyunEmbeddings(), # 用AliyunEmbeddings Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name
print("3.Save chunked_documents to vectorstore!")

# 4. Retrieval 准备模型和Retrieval链
# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 创建ChatOpenAI实例，指定模型名称、API Key和base_url
# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)
print("4.Created Retrieval !")
# 5. Output 问答系统的UI实现
app = Flask(__name__)  # Flask APP
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')

        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain.invoke({"query": question})

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
    print("5.Created APP!")