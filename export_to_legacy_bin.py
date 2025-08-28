# export_to_legacy_bin.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from record_activations import load_model_and_tokenizer

# src = "sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft"   # 你的训练输出目录
# dst = "export_for_vllm/llama2-13b-math-code-alignment-sparseft"                  # 新导出目录

# src = "dense_ft_ckpts/llama2-13b-math-code-alignment-denseft"
# dst = "export_for_vllm/llama2-13b-math-code-alignment-denseft"

# src = "sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft/epoch1_batch_size4_grad_accum16_step24_tok28k"   # 你的训练输出目录
# dst = "export_for_vllm/llama2-13b-math-code-alignment-sparseft/epoch1_batch_size4_grad_accum16_step24_tok28k" 

# src = "sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft/epoch1_batch_size4_grad_accum16_step48_tok55k"   # 你的训练输出目录
# dst = "export_for_vllm/llama2-13b-math-code-alignment-sparseft/epoch1_batch_size4_grad_accum16_step48_tok55k" 

src = "sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft/epoch0_batchSize1_gradAccum16_ckpt-step1024"   # 你的训练输出目录
dst = "export_for_vllm/sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft/epoch0_batchSize1_gradAccum16_ckpt-step1024" 



os.makedirs(dst, exist_ok=True)

# 1) 加载到 CPU；旧环境一般也能 load
model = AutoModelForCausalLM.from_pretrained(
    src,
    # torch_dtype=torch.float16,     # 或 "auto"；想省空间就 fp16
    device_map="cpu",
    # low_cpu_mem_usage=True,
)
tok = AutoTokenizer.from_pretrained(src, use_fast=False)

# model, tok = load_model_and_tokenizer("Llama-2-13b-hf", half_model_dtype=False, seed=0, device="auto")


# 2) 保存为 pytorch_model-*.bin（很多旧 vLLM 版本最稳）
model.save_pretrained(dst)
tok.save_pretrained(dst)

print("Done ->", dst)