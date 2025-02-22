from llama_cpp import Llama
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from pydantic import Field

# 模型的名称和路径常量
# https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/tree/main
MODEL_NAME = 'llava-llama-3-8b-v1_1-int4.gguf'
MODEL_PATH = r'E:\huggingface\download'  # 使用原始字符串避免转义问题


class CustomLLM(LLM):
    model_name: str = MODEL_NAME  # 直接定义为实例变量，除非确实需要类变量特性
    llm: Llama = Field(...)  # 定义 llm 字段

    def __init__(self, **kwargs):
        llama = Llama(model_path=f"{MODEL_PATH}\\{MODEL_NAME}", n_threads=4)
        super().__init__(llm=llama, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.llm(f"Q: {prompt} A: ", max_tokens=256)
        output = response['choices'][0]['text'].replace('A: ', '').strip()
        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

    # 如果invoke是推荐的方式，则可以这样定义invoke方法
    def invoke(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)


# 初始化自定义LLM类
llm = CustomLLM()

# 使用自定义LLM生成一个回复
result = llm.invoke("写一个关于玫瑰花的故事")

# 打印生成的回复
print(result)