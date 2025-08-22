# export_to_legacy_bin.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# src = "sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft"   # 你的训练输出目录
# dst = "export_for_vllm/llama2-13b-math-code-alignment-sparseft"                  # 新导出目录

src = "dense_ft_ckpts/llama2-13b-math-code-alignment-denseft"
dst = "export_for_vllm/llama2-13b-math-code-alignment-denseft"


os.makedirs(dst, exist_ok=True)

# 1) 加载到 CPU；旧环境一般也能 load
model = AutoModelForCausalLM.from_pretrained(
    src,
    # torch_dtype=torch.float16,     # 或 "auto"；想省空间就 fp16
    device_map="cpu",
    # low_cpu_mem_usage=True,
)
tok = AutoTokenizer.from_pretrained(src, use_fast=False)

# 2) 保存为 pytorch_model-*.bin（很多旧 vLLM 版本最稳）
model.save_pretrained(dst)
tok.save_pretrained(dst)

print("Done ->", dst)