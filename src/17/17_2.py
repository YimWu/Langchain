import os
from langchain_community.utilities import SQLDatabase  # 更新后的导入路径
from langchain_core.caches import BaseCache
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import Callbacks  # 导入 Callbacks
import re

# 定义 BaseCache 类，这里使用一个简单的空实现作为示例
class CustomCache(BaseCache):
    def lookup(self, prompt, llm_string):
        return None

    def update(self, prompt, llm_string, return_val):
        pass

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

# 设置你的API Key和基础URL，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建ChatOpenAI实例
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=1
)

# 定义 BaseCache 并重建模型
SQLDatabaseChain.model_rebuild()

# 创建SQL数据库链实例
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 查询示例
queries = [
    # "有多少种不同的鲜花？",
    # "哪种鲜花的存货数量最少？",
    # "平均销售价格是多少？",
    # "从法国进口的鲜花有多少种？",
    "哪种鲜花的销售量最低？",
    "哪种鲜花的销售量最高？"
]

def clean_sql(sql):
    # 去除 Markdown 代码块标记
    return re.sub(r'```(sql)?\s*|\s*```', '', sql).strip()

for query in queries:
    try:
        response = db_chain.invoke(query)
        # 检查响应中是否包含 SQL 查询
        if isinstance(response, dict) and 'query' in response.get('result', ''):
            # 提取 SQL 查询并清理
            sql_match = re.search(r'SQLQuery:\s*(.*)', response['result'])
            if sql_match:
                sql = sql_match.group(1)
                clean_sql_query = clean_sql(sql)
                # 重新执行清理后的 SQL 查询
                result = db.database.invoke(clean_sql_query)
                response = {'query': query, 'result': f'SQLQuery: {clean_sql_query}\nSQLResult: {result}'}
        print(f"Query: {query}\nResponse: {response}")
    except Exception as e:
        print(f"Query: {query}\nError: {e}")
