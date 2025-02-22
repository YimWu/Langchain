import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time

# 记录开始时间
start_time = time.time()

# 检查CUDA是否可用
print("Is CUDA available:", torch.cuda.is_available())
step1_time = time.time()
print(f"检查CUDA可用性耗时: {step1_time - start_time:.2f} 秒")

# 加载环境变量
huggingface_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# 指定路径，运行时自动通过该国内路径下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=huggingface_key)
step2_time = time.time()
print(f"加载分词器耗时: {step2_time - step1_time:.2f} 秒")

# 加载预训练的模型，明确指定使用CUDA设备，并使用混合精度
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map=device,
    token=huggingface_key,
    torch_dtype=torch.float16  # 使用混合精度
)
step3_time = time.time()
print(f"加载模型耗时: {step3_time - step2_time:.2f} 秒")

# 如果模型配置中没有设置pad_token_id，则使用eos_token_id作为pad_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

# 定义一个提示，希望模型基于此提示生成故事
prompt = "你是什么模型"

# 使用分词器将提示转化为模型可以理解的格式，并创建attention mask
inputs = tokenizer(prompt, return_tensors="pt")
step4_time = time.time()
print(f"处理输入提示耗时: {step4_time - step3_time:.2f} 秒")

# 将输入移动到相同的设备上
inputs = {k: v.to(device) for k, v in inputs.items()}
step5_time = time.time()
print(f"将输入移动到设备耗时: {step5_time - step4_time:.2f} 秒")

print("开始生成文本...")
# 使用模型生成文本，同时传递attention_mask
with torch.cuda.amp.autocast():  # 使用自动混合精度
    outputs = model.generate(inputs["input_ids"],
                             attention_mask=inputs["attention_mask"],
                             max_new_tokens=100)  # 进一步减少max_new_tokens的值
step6_time = time.time()
print(f"文本生成耗时: {step6_time - step5_time:.2f} 秒")
print("文本生成完成。")

# 将生成的令牌解码成文本，并跳过任何特殊的令牌
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
step7_time = time.time()
print(f"解码生成文本耗时: {step7_time - step6_time:.2f} 秒")

# 打印生成的响应
print(response)
total_time = time.time() - start_time
print(f"整个流程总耗时: {total_time:.2f} 秒")