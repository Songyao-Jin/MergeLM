import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.load_config import cache_dir

model_name = "Llama-2-13b-hf"

# 检查并创建保存结构目录
os.makedirs("model_structure", exist_ok=True)
model_path = os.path.join(cache_dir, model_name)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

# 打印并保存模型结构
structure_path = f"model_structure/{model_name}_structure.txt"
with open(structure_path, "w") as f:
    f.write(str(model))

print(f"Model structure saved to {structure_path}")