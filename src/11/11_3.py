from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
import os

# 初始化大语言模型
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.2
)
# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

# 第一天的对话
# 回合1
result = conversation.invoke("我姐姐明天要过生日，我需要一束生日花束。")
print("第一次对话后的记忆:", result)

# 回合2
result = conversation("她喜欢粉色玫瑰，颜色是粉色的。")

print("第二次对话", result)

# 第二天的对话
# 回合3
result = conversation("我又来了，还记得我昨天为什么要来买花吗？")
print("第三次对话", result)