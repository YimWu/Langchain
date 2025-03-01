import os

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import asyncio

# 初始化大语言模型
# 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 创建ChatOpenAI实例，指定模型名称、API Key和base_url
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.5
)
# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# 使用context manager进行token counting
with get_openai_callback() as cb:
    # 第一天的对话
    # 回合1
    conversation.invoke("我姐姐明天要过生日，我需要一束生日花束。")
    print("第一次对话后的记忆:", conversation.memory.buffer)

    # 回合2
    conversation.invoke("她喜欢粉色玫瑰，颜色是粉色的。")
    print("第二次对话后的记忆:", conversation.memory.buffer)

    # 回合3 （第二天的对话）
    conversation.invoke("我又来了，还记得我昨天为什么要来买花吗？")
    print("/n第三次对话后时提示:/n",conversation.prompt.template)
    print("/n第三次对话后的记忆:/n", conversation.memory.buffer)

# 输出使用的tokens
print("\n总计使用的tokens:", cb.total_tokens)

async def additional_interactions():
    with get_openai_callback() as cb:
        tasks = []
        for _ in range(3):
            # 使用HumanMessage包装输入
            message = HumanMessage(content="我姐姐喜欢什么颜色的花？")
            task = llm.agenerate([[message]])
            tasks.append(task)
        await asyncio.gather(*tasks)
    print("\n另外的交互中使用的tokens:", cb.total_tokens)

# 运行异步函数
# 手动管理事件循环
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(additional_interactions())
    finally:
        loop.close()

