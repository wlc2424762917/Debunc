from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer

# 指定模型名称或路径
model_name = "/data/hf_models/Llama-3.1-8B-Instruct"  # 替换为实际的模型名称或路径

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")

print(model)
# 如果需要，可以将模型移动到 GPU
# model.to('cuda')