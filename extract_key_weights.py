#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按层提取关键权重（Top-r% 到 Top-l% 区间），输出 mask 字典。
用法示例：
python extract_key_weights.py \
  --ref_model_path base_state_dict.pth \
  --trained_model_path finetuned_state_dict.pth \
  --out_mask_path mask_top5_10.pt \
  --l_per 5 --r_per 10 \
  --device cpu  # 或 cuda:0

说明：
- 仅当 r_per >= l_per 且二者在 [0,100] 内有效
- 区间含义：选择“前 r% 以内但不超过前 l% 的权重”
  例：l=5, r=10 -> 选择 90~95 分位之间（Top10%~Top5%）的权重
"""


import os
# 🔐 Redirect all Hugging Face-related caches and locks to your personal directory
os.environ["HF_HOME"] = "/data/songyao/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/songyao/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/songyao/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/data/songyao/.cache/huggingface/hub"



import math
import argparse
from typing import Dict
import torch
from record_activations import load_model_and_tokenizer



# def load_state_dict(model) -> Dict[str, torch.Tensor]:
#     # 简洁版：假设传入的是 .pth/.pt/.bin 的 state_dict
#     # 若你要支持 HF 目录，可改成 AutoModelForCausalLM.from_pretrained(...).state_dict()
#     # sd = torch.load(path, map_location="cpu")
#     if hasattr(model, "state_dict"):
#         sd = model.state_dict()
#     return sd


@torch.no_grad()
def extract_key_weight_masks(
    ref_sd: Dict[str, torch.Tensor],
    ft_sd: Dict[str, torch.Tensor],
    l_per: float,
    r_per: float,
    device: str = "cpu",
    include_bias: bool = False,
    include_norm: bool = True,
    include_embed: bool = True,
    only_weight_params: bool = True,
    mask_dtype: torch.dtype = torch.bfloat16,  # 后续稀疏训练常用 0/1 bfloat16
):
    """
    返回 mask_dict: {name: Tensor(0/1, same shape)}，并统计覆盖率
    """
    assert 0 <= l_per <= 100 and 0 <= r_per <= 100, "percent 必须在 [0,100]"
    assert r_per >= l_per, "需要满足 r_per >= l_per（Top-r% 到 Top-l%）"

    mask_dict = {}
    total_elems = 0
    total_masked = 0

    # 统一遍历 ft_sd 的键（只处理两边都存在且 shape 一致的）
    keys = [k for k in ft_sd.keys() if (k in ref_sd and ft_sd[k].shape == ref_sd[k].shape)]

    for k in keys:
        
        print("The weight matrix we deal with is: ",k)
        
        # 可选：仅处理 .weight
        if only_weight_params and (not k.endswith(".weight")):
            continue

        # 过滤 norm / embed / bias（按需改）
        lk = k.lower()
        if (not include_bias) and lk.endswith(".bias"):
            continue
        if (not include_norm) and ("layernorm" in lk or "rmsnorm" in lk or ".norm" in lk):
            continue
        if (not include_embed) and ("embed" in lk):
            continue

        base = ref_sd[k]
        ft = ft_sd[k]
        if not torch.is_floating_point(base) or not torch.is_floating_point(ft):
            print(f"{k} is not float.")
            continue  # 跳过非浮点参数

        # 本参数的 |Δ|
        delta_abs = (ft - base).abs().to(device)

        n = delta_abs.numel()
        if n == 0:
            continue

        # 需要的数量（向上取整保证至少取到一个）
        k_high = int(math.ceil(n * (r_per / 100.0)))  # Top-r%
        k_low  = int(math.ceil(n * (l_per / 100.0)))  # Top-l%

        # 将 1D 视图拿来做 k-th（kthvalue 返回第 k 小；我们要“第 k 大”的阈值）
        flat = delta_abs.view(-1)

        # 边界处理：
        # - r_per == 0：不选任何（阈值设为 +inf，mask 全 0）
        # - l_per == 0：上界阈值 = +inf（表示不排除最顶层的部分）
        INF = torch.tensor(float("inf"), device=device)

        if r_per == 0:
            thr_high = INF  # 使 (val >= thr_high) 恒为 False
        else:
            # 第 k_high 大 <=> 第 (n - k_high + 1) 小
            idx_small = max(1, n - k_high + 1)
            thr_high = flat.kthvalue(idx_small).values  # 标量。#进入 Top-r% 的最小值门槛。

        if l_per == 0:
            thr_low = INF  # 不排除 Top-l%（因为 l=0）
        else:
            idx_small = max(1, n - k_low + 1)
            thr_low = flat.kthvalue(idx_small).values   #进入 Top-l% 的最小值门槛。

        # 目标区间： >= thr_high 且 < thr_low
        # 例：l=5%, r=10% -> 选 Top10% 到 Top5% 之间
        if r_per == 0 or delta_abs.max() == 0:
            local_mask = torch.zeros_like(delta_abs, dtype=torch.bool, device=device)
        else:
            if thr_high == 0:
                cond_high = delta_abs > thr_high
            else:
                cond_high = delta_abs >= thr_high
            if l_per > 0:
                cond_low_exclude = delta_abs >= thr_low  # 这些属于 Top-l%，需要排除
                local_mask = cond_high & (~cond_low_exclude)
            else:
                local_mask = cond_high

        # 统计 & 存储
        total_elems += n
        total_masked += local_mask.sum().item()

        # 保存为指定 dtype（0/1）
        mask_val = local_mask.to(mask_dtype)
        # 回 CPU 存盘更通用
        mask_dict[k] = mask_val.cpu()

    ratio = 100.0 * (total_masked / max(1, total_elems))
    return mask_dict, ratio



def _sanitize(name: str) -> str:
    """把模型名里的 / \ 空格 等替换成 _，用于安全的文件名/目录名。"""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

def build_out_mask_path(
    base_model_name: str,
    ft_model_name: str,
    l_per: float,
    r_per: float,
    include_bias: bool,
    include_norm: bool,
    include_embed: bool,
    only_weight_params: bool,
    mask_dtype: str = "bfloat16",
    root_dir: str = "masks"
) -> str:
    """
    根据模型名与筛选参数自动生成输出路径:
      masks/{base}__{ft}/top{r}_to_top{l}__flags__{dtype}.pt
    例如:
      masks/meta-llama_Llama-2-13b-hf__WizardLM_WizardMath-13B-V1.0/top10_to_top0__bias-norm-embed__bf16.pt
    """
    b = _sanitize(base_model_name)
    f = _sanitize(ft_model_name)
    flags = []
    if include_bias:  flags.append("bias")
    if include_norm:  flags.append("norm")
    if include_embed: flags.append("embed")
    if only_weight_params: flags.append("onlyW")
    flags_str = "-".join(flags) if flags else "all"
    dtype_short = {"bool":"bool", "float16":"fp16", "bfloat16":"bf16", "float32":"fp32"}[mask_dtype]
    subdir = os.path.join(root_dir, f"{b}_{f}")
    fname = f"top{int(r_per)}_to_top{int(l_per)}_{flags_str}_{dtype_short}.pt"
    return os.path.join(subdir, fname)



def main():
    
    base_model_name = "Llama-2-13b-hf"
    ft_model_name =  "llama-2-13b-code-alpaca"           #"WizardLM-13B-V1.2"        #"llama-2-13b-code-alpaca"           #"WizardMath-13B-V1.0"
    l_per = 0.0               # Top-l%（上界，0 表示不排除最顶层）
    r_per = 10.0              # Top-r%（下界）
    device = "cpu"            # 建议用 cpu 以避免多卡 device mismatch
    include_bias = True
    include_norm = True
    include_embed = True
    only_weight_params = False
    mask_dtype = "bfloat16"   # "bool"|"float16"|"bfloat16"|"float32"
    
    
    dtype_map = {
        "bool": torch.bool,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }


    out_mask_path = build_out_mask_path(
        base_model_name, ft_model_name, l_per, r_per,
        include_bias, include_norm, include_embed, only_weight_params,
        mask_dtype=mask_dtype, root_dir="masks"
    )

    # ===== 加载模型 =====
    print("加载 base, finetuned 模型 ...")
    base_model, _ = load_model_and_tokenizer(base_model_name, half_model_dtype=False, seed=0)
    ft_model, _ = load_model_and_tokenizer(ft_model_name, half_model_dtype=False, seed=0)


    # ===== 2. 取 state_dict 并统一搬到 CPU，避免 (ft - base) 的跨设备问题 =====
    print("抽取 state_dict 到 CPU ...")
    ref_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    ft_sd  = {k: v.detach().cpu() for k, v in ft_model.state_dict().items()}
    

    # ===== 3. 提取关键权重 =====
    print(f"开始按层提取关键权重: Top-{r_per}% 到 Top-{l_per}%")
    mask_dict, ratio = extract_key_weight_masks(
        ref_sd, ft_sd,
        l_per=l_per, r_per=r_per,
        device=device,  # 这里依然传 "cpu" 就好
        include_bias=include_bias,
        include_norm=include_norm,
        include_embed=include_embed,
        only_weight_params=only_weight_params,
        mask_dtype=dtype_map[mask_dtype],
    )

    # ===== 4. 保存 =====
    os.makedirs(os.path.dirname(out_mask_path) or ".", exist_ok=True)
    torch.save(mask_dict, out_mask_path)
    print(f"已保存 mask 到: {out_mask_path}")
    print(f"Mask 覆盖率: {ratio:.2f}%")

if __name__ == "__main__":
    main()
