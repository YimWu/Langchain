import asyncio
import os
from typing import Any, Dict, List
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_openai import ChatOpenAI


# 创建同步回调处理器
class MyFlowerShopSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"获取花卉数据: token: {token}")

# 创建异步回调处理器
class MyFlowerShopAsyncHandler(AsyncCallbackHandler):

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("正在获取花卉数据...")
        await asyncio.sleep(0.5)  # 模拟异步操作
        print("花卉数据获取完毕。提供建议...")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("整理花卉建议...")
        await asyncio.sleep(0.5)  # 模拟异步操作
        print("祝你今天愉快！")

# 主要的异步函数
async def main():
    # 设置你的API Key和基础URL，这里假设你使用的是阿里云的服务，请根据实际情况替换
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 创建ChatOpenAI实例，指定模型名称、API Key和base_url
    flower_shop_chat = ChatOpenAI(
        model_name="qwen-plus",
        openai_api_key=api_key,
        openai_api_base=base_url,
        max_tokens=100,
        streaming=True,
        callbacks=[MyFlowerShopSyncHandler(), MyFlowerShopAsyncHandler()],
    )

    # 异步生成聊天回复
    await flower_shop_chat.agenerate([[HumanMessage(content="哪种花卉最适合生日？只简单说3种，不超过50字")]])

# 运行主异步函数
asyncio.run(main())