import os
# 🔐 Redirect all Hugging Face-related caches and locks to your personal directory
os.environ["HF_HOME"] = "/data/songyao/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/songyao/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/songyao/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/data/songyao/.cache/huggingface/hub"



# sparse_finetune.py
import os, math, json, random
from typing import Dict, Iterable, Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import wandb
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from record_activations import load_model_and_tokenizer
from extract_key_weights import _sanitize
from utils.utils import set_random_seed
import bitsandbytes as bnb


# ---------------- 逐参数平均 merge delta  ----------------
def build_dense_avg_delta(
    base_model_name_or_path: str,
    ft_model_names_or_paths: List[str],
    out_dense_delta_path: str = "dense_merged_delta_avg.pt",
    seed = 0,
) -> Dict[str, torch.Tensor]:
    """
    对多个 FT 模型的 delta 做逐参数平均：delta_avg[k] = mean_i( ft_i[k] - base[k] )
    """
    
    base_model, _ = load_model_and_tokenizer(base_model_name_or_path, half_model_dtype=False, seed=seed)
    base_sd = base_model.state_dict()
    del base_model
    
    ft_sds=[]
    for ft_name in ft_model_names_or_paths:
        ft_model, _ = load_model_and_tokenizer(ft_name, half_model_dtype=False, seed=seed)
        ft_sd = ft_model.state_dict()
        ft_sds.append(ft_sd)
        del ft_model
    
    
    merged_delta: Dict[str, torch.Tensor] = {}
    for k, v_base in base_sd.items():
        vs = []
        for sd in ft_sds:
            v_ft = sd.get(k, None)
            if v_ft is not None and v_ft.shape == v_base.shape:
                vs.append(v_ft)
        if not vs:
            continue
        acc = None
        for v_ft in vs:
            d = (v_ft - v_base)  # 统一到 fp32 做加法更稳
            acc = d if acc is None else acc + d
        merged_delta[k] = (acc / len(vs)).to(v_base.dtype)

    os.makedirs(os.path.dirname(out_dense_delta_path) or ".", exist_ok=True)
    torch.save(merged_delta, out_dense_delta_path)
    print(f"[OK] dense merged delta 已保存到: {out_dense_delta_path}")
    return merged_delta
# ------------------------------------------------


# ---------------- extra train layers  ----------------
def _infer_mask_dtype(mask: Dict[str, torch.Tensor]) -> torch.dtype:
    for v in mask.values():
        if torch.is_tensor(v):
            return v.dtype
    return torch.float32  # 兜底


def promote_layers_in_mask(
    mask: Dict[str, torch.Tensor],
    model,
    layers_to_force: List[int],
    mask_dtype: Optional[torch.dtype] = None,
    include_patterns: Optional[List[str]] = None,   # 为空则所有参数都放开
):
    """
    将 model.layers.{i}.** 下的参数 mask 置为全 1（或 True），使这些层参与训练。
    - mask: 现有的 name->tensor 掩码（CPU 上）
    - layers_to_force: 例如 [0,1,2,3,4,35,36,37,38,39]
    - include_patterns: 想只放开某些子模块时用，比如 ["self_attn", "mlp"]，默认 None 表示该层所有参数
    """
    if mask_dtype is None:
        mask_dtype = _infer_mask_dtype(mask)

    is_bool = (mask_dtype == torch.bool)

    def _need_param(name: str, layers_to_force, include_patterns) -> bool:
        ok = any(f"model.layers.{i}." in name for i in layers_to_force)
        if not ok:
            return False
        if include_patterns is None:
            return True
        return any(pat in name for pat in include_patterns)

    changed, added = 0, 0
    for name, p in model.named_parameters():
        if not _need_param(name, layers_to_force, include_patterns):
            continue
        full = torch.ones_like(p, dtype=mask_dtype, device="cpu")
        if is_bool:
            full = full.bool()
        if name in mask:
            # 与原 mask 合并（布尔 OR / 浮点 max）
            old = mask[name].to(dtype=mask_dtype, device="cpu")
            if is_bool:
                mask[name] = (old.bool() | full.bool())
            else:
                mask[name] = torch.maximum(old, full)
            changed += 1
        else:
            mask[name] = full
            added += 1

    print(f"[promote_layers_in_mask] 覆盖/合并: {changed} 个参数，新增: {added} 个参数")
    return mask

# ------------------------------------------------


# ---------------- 辅助函数 ----------------
def apply_delta_inplace(model: torch.nn.Module, delta: Dict[str, torch.Tensor]):
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in delta.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].add_(v.to(sd[k].dtype).to(sd[k].device))

# def register_grad_masks(model: torch.nn.Module, mask: Dict[str, torch.Tensor]):
#     """为参与训练的参数注册梯度掩码；mask 张量需与 param.shape 相同（bool/float均可）"""
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         m = mask.get(name, None)
#         if m is None:
#             # 没有显式 mask 的参数：保持梯度为 0（等价于不训练）
#             p.register_hook(lambda g: torch.zeros_like(g))
#         else:
#             m = m.to(dtype=p.dtype, device=p.device)
#             p.register_hook(lambda g, m=m: g * m)
            
#             # def make_hook(m):
#             #     def hook(g):
#             #         return g * m
#             #     return hook
#             # p.register_hook(make_hook(m))


def apply_grad_masks_step_(model: torch.nn.Module, mask: Dict[str, torch.Tensor]):
    """
    在 optimizer.step() 之前调用：
    把每个 param.grad 乘以对应 mask（mask 常驻 CPU；此处临时搬到 grad.device，用完即丢）
    对没有 mask 或 mask 全 0 的参数，将 grad 清零，保证它们不更新。
    """
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        m = mask.get(name, None)
        if m is None:
            # 没有显式 mask：不训练
            p.grad.detach().zero_()
            continue
        # 兼容非 bool 存储
        if m.dtype != torch.bool:
            m = (m != 0)
        if not torch.any(m):
            p.grad.detach().zero_()
            continue
        # 临时搬到梯度所在设备
        gm = m.to(device=p.grad.device, dtype=p.grad.dtype, non_blocking=True)
        p.grad.mul_(gm)   # 原地掩码
        # gm 会在本次循环结束后被释放，不常驻显存


# ---------------- 数据集封装（Math） ----------------
class MathSFTDataset(Dataset):
    def __init__(self, tokenizer, max_len=2048, gsm8k_n=None, math_n=None):
        self.tok = tokenizer
        self.samples = []

        # GSM8K（train）
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train")  # Q & A（带步骤）
        if gsm8k_n: ds_gsm = ds_gsm.select(range(min(gsm8k_n, len(ds_gsm))))
        # ds_gsm = load_gsm8k_train(max_n=gsm8k_n, subset="main")
        for ex in ds_gsm:
            q = ex["question"].strip()
            a = ex["answer"].strip()
            prompt = f"Question: {q}\n\nAnswer (step-by-step):\n"
            target = a
            self.samples.append((prompt, target))

        # MATH（train）
        ds_math = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
        if math_n: ds_math = ds_math.select(range(min(math_n, len(ds_math))))
        for ex in ds_math:
            q = ex["problem"].strip()
            sol = ex["solution"].strip()
            prompt = f"Question: {q}\n\nAnswer (step-by-step):\n"
            target = sol
            self.samples.append((prompt, target))

        random.shuffle(self.samples)
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        # 只对 target 计 loss：prompt 部分 label = -100
        text = prompt + target
        enc_all = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        enc_prompt = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_len)
        input_ids = enc_all.input_ids[0]
        labels = input_ids.clone()
        labels[:enc_prompt.input_ids.shape[1]] = -100
        return {"input_ids": input_ids, "labels": labels, "attention_mask": enc_all.attention_mask[0]}


# ---------------- 数据集封装（Code Alpaca） ----------------
class CodeAlpacaDataset(Dataset):
    def __init__(self, tokenizer, max_len=2048, n=None):
        self.tok = tokenizer
        ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
        if n: ds = ds.select(range(min(n, len(ds))))
        self.samples = []
        for ex in ds:
            instr = (ex.get("instruction") or "").strip()
            inp   = (ex.get("input") or "").strip()
            out   = (ex.get("output") or "").strip()
            prompt = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n"
            target = out
            self.samples.append((prompt, target))
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        text = prompt + target
        enc_all = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        enc_prompt = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_len)
        input_ids = enc_all.input_ids[0]
        labels = input_ids.clone()
        labels[:enc_prompt.input_ids.shape[1]] = -100
        return {"input_ids": input_ids, "labels": labels, "attention_mask": enc_all.attention_mask[0]}



# ---------------- 数据集封装（Evol-Instruct V2） ----------------
class EvolInstructV2Dataset(Dataset):
    """
    默认加载 WizardLMTeam/WizardLM_evol_instruct_V2_196k
    兼容字段名：instruction / input / output|response|answer
    训练时只对 Response 段计 loss（Prompt 段 label = -100）
    """
    def __init__(self, tokenizer, dataset_name="WizardLMTeam/WizardLM_evol_instruct_V2_196k",
                 split="train", max_len=2048, n=None):
        self.tok = tokenizer
        ds = load_dataset(dataset_name, split=split)
        if n: ds = ds.select(range(min(n, len(ds))))
        self.samples = []

        for ex in ds:
            instr = (ex.get("instruction") or "").strip()
            inp   = (ex.get("input") or "").strip()
            # 不同版本字段名可能不同，做个兜底
            out   = (ex.get("output") or ex.get("response") or ex.get("answer") or "").strip()

            # Alpaca/WizardLM 常见格式
            prompt = f"### Instruction:\n{instr}\n"
            if len(inp) > 0:
                prompt += f"### Input:\n{inp}\n"
            prompt += "### Response:\n"

            target = out
            if target == "":   # 空样本跳过
                continue
            self.samples.append((prompt, target))

        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        text = prompt + target
        enc_all    = self.tok(text,   return_tensors="pt", truncation=True, max_length=self.max_len)
        enc_prompt = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_len)
        input_ids = enc_all.input_ids[0]
        labels    = input_ids.clone()
        labels[:enc_prompt.input_ids.shape[1]] = -100  # 只训练 response
        return {"input_ids": input_ids, "labels": labels, "attention_mask": enc_all.attention_mask[0]}


def collate(batch, pad_id):
    ids = [b["input_ids"] for b in batch]
    ams = [b["attention_mask"] for b in batch]
    lbs = [b["labels"] for b in batch]
    maxlen = max(x.size(0) for x in ids)
    def pad_stack(xs, val=0):
        out = []
        for x in xs:
            if x.size(0) < maxlen:
                pad = torch.full((maxlen - x.size(0),), val, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            out.append(x)
        return torch.stack(out, dim=0)
    return {
        "input_ids": pad_stack(ids, pad_id),
        "attention_mask": pad_stack(ams, 0),
        "labels": pad_stack(lbs, -100),
    }
    
    
    
def sparse_finetune(
    base_model_name_or_path: str,
    # --- 稀疏路径（对于sparse merge和train） ---
    merged_sparse_delta_path: str = "dummy",
    merged_mask_path: str = "dummy",
    # --- 新增：全参训练(dense) ---
    train_all_weights: bool = False,                       # ← True 走全参
    dense_merged_delta_path: str = None,           # ← 平均后的 dense delta 路径
    # --- 其它原有参数 ---
    out_dir: str = "dummy",
    task_mix: dict = {"math": 1.0, "code": 1.0, "general": 0.0},   # ← 新增 general
    math_sizes=(8000, 8000),
    code_n=50000,
    general_n=30000,                                               # ← 新增：EvolV2 采样数
    # evol_ds_name="WizardLMTeam/WizardLM_evol_instruct_V2_196k",    # ← 新增：数据集名
    lr=1e-5,
    epochs=1,
    batch_size=1,
    grad_accum=16,
    # bf16=True,
    wandb_proj="sparse-ft",
    run_name="sparse_ft_run",
    seed=42,
    # 可选：给全参/稀疏不同的 weight_decay
    wd_sparse: float = 0.0, wd_dense: float = 0.01,
    # ↓↓↓ 新增：想额外放开的层索引
    extra_train_layers: Optional[List[int]] = None,
    extra_patterns: Optional[List[str]] = None,  # 例如只放开 ["self_attn","mlp"]
):
    set_random_seed(seed)
    # 1) 加载模型
    print("load")
    model, tok = load_model_and_tokenizer(base_model_name_or_path, half_model_dtype=False, seed=seed, device="balanced_low_0")
    print("load successfully")
    model.train()
    
    # 2) 应用 merged 稀疏 delta, 或者全参数delta
    if train_all_weights:
        assert dense_merged_delta_path is not None, "train_all=True 需提供 dense_merged_delta_path"
        dense_delta = torch.load(dense_merged_delta_path, map_location="cpu")
        apply_delta_inplace(model, dense_delta)
    else:
        assert merged_sparse_delta_path is not None and merged_mask_path is not None, \
            "稀疏训练必须提供 merged_sparse_delta_path 和 merged_mask_path"
        merged_delta = torch.load(merged_sparse_delta_path, map_location="cpu")
        apply_delta_inplace(model, merged_delta)
        
        # 3) 注册梯度掩码
        merged_mask = torch.load(merged_mask_path, map_location="cpu")
        # register_grad_masks(model, merged_mask)
        
        # ★ 将指定层的 mask 置为全 1
        if extra_train_layers:
            merged_mask = promote_layers_in_mask(
                merged_mask, model,
                layers_to_force=extra_train_layers,
                mask_dtype=None,                 # 自动跟现有 mask 的 dtype
                include_patterns=extra_patterns  # None=整层，或 ["self_attn","mlp"]
            )
    
    
    
    
    # 4) 数据
    datasets: List[Dataset] = []
    if task_mix.get("math", 0) > 0:
        ds_math = MathSFTDataset(tok, max_len=1024, gsm8k_n=math_sizes[0], math_n=math_sizes[1])
        datasets.append(ds_math)

    if task_mix.get("code", 0) > 0:
        ds_code = CodeAlpacaDataset(tok, max_len=1024, n=code_n)
        datasets.append(ds_code)

    # 新增：Evol-Instruct V2（通用指令）
    if task_mix.get("general", 0) > 0:
        ds_gen = EvolInstructV2Dataset(tok, max_len=1024, n=general_n)
        datasets.append(ds_gen)
        
    
    # 简单拼接（需要更精细采样时可写 WeightedRandomSampler）
    train_ds = ConcatDataset(datasets)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id)
    )
    
    
    # 5) 优化器 & 调度
    if train_all_weights:
        weight_decay = wd_dense
    else:
       weight_decay = wd_sparse
    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / (batch_size * grad_accum))
    num_steps = steps_per_epoch * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * num_steps), num_steps)
    
    # 6) W&B
    if wandb_proj:
        wandb.init(project=wandb_proj, name=run_name, config={
            "base": base_model_name_or_path,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "grad_accum": grad_accum,
            "math_sizes": math_sizes, "code_n": code_n, "general_n": general_n,
        })
        
    # 7) 训练
    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            loss = out.loss / grad_accum
            loss.backward()
            running += loss.item()
            
            if (i + 1) % grad_accum == 0:
                if train_all_weights == False:
                    # 在 step 前做一次“步前梯度掩码”，避免 mask 常驻 GPU
                    apply_grad_masks_step_(model, merged_mask)  # 稀疏才需要
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                sch.step()
                global_step += 1
                if wandb_proj and global_step % 10 == 0:
                    wandb.log({"loss": running, "lr": sch.get_last_lr()[0], "step": global_step})
                running = 0.0

        print(f"Epoch {epoch+1} done.")    
        
    # 8) 保存 HF 目录（可被 vLLM/Transformers 直接加载）
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] Saved to: {out_dir}")
    if wandb_proj: 
        wandb.finish()
            
            
            

def main():
    
    # # sparse finetune
    # sparse_finetune(
    #     base_model_name_or_path="Llama-2-13b-hf",
    #     merged_sparse_delta_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_sparse_delta_union_avg_top10.pt",
    #     merged_mask_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_mask_union_top10.pt",
    #     out_dir="sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft",
    #     task_mix={"math":1.0, "code":1.0, "general": 1.0},   # 做数学+代码+对齐
    #     math_sizes=(8000, 8000),             # GSM8K 8k + MATH 8k（可按显存调小）
    #     code_n=8000,                        # CodeAlpaca 子集
    #     general_n=8000,
    #     lr=1e-5, epochs=1, batch_size=1, grad_accum=16,
    #     wandb_proj="weight-sparse-ft", run_name="llama2-13b-sparseft"
    #     )
    
    # # densely all weights finetune
    # build_dense_avg_delta(
    #     base_model_name_or_path = "Llama-2-13b-hf",
    #     ft_model_names_or_paths = ["WizardLM-13B-V1.2", "llama-2-13b-code-alpaca", "WizardMath-13B-V1.0"],
    #     out_dense_delta_path = "dense_merge_outputs/llama2_13b_avg_three.pt",
    #     seed = 42,
    #     )
    # sparse_finetune(
    #     base_model_name_or_path="Llama-2-13b-hf",
    #     train_all_weights=True,
    #     dense_merged_delta_path="dense_merge_outputs/llama2_13b_avg_three.pt",
    #     # 稀疏相关参数可不传
    #     out_dir="dense_ft_ckpts/llama2-13b-math-code-alignment-denseft",
    #     task_mix={"math":1.0, "code":1.0, "general":1.0},
    #     math_sizes=(8000, 8000), code_n=8000, general_n=8000,
    #     lr=1e-5, epochs=1, batch_size=1, grad_accum=16,
    #     wd_dense=0.01,
    #     wandb_proj="wanda-dense-ft", run_name="llama2-13b-denseft",
    # )
    
    
    # sparse finetune with extra_train_layers
    sparse_finetune(
        base_model_name_or_path="Llama-2-13b-hf",
        merged_sparse_delta_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_sparse_delta_union_avg_top10.pt",
        merged_mask_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_mask_union_top10.pt",
        out_dir="sparse_ft_ckpts/llama2-13b-math-code-alignment-estra_train_layers_[0,1,2,3,4,35,36,37,38,39]-sparseft",
        task_mix={"math":1.0, "code":1.0, "general": 1.0},   # 做数学+代码+对齐
        math_sizes=(8000, 8000),             # GSM8K 8k + MATH 8k（可按显存调小）
        code_n=8000,                        # CodeAlpaca 子集
        general_n=8000,  25steps
        lr=1e-5, epochs=1, batch_size=1, grad_accum=16,
        wandb_proj="weight-sparse-ft", run_name="llama2-13b-sparseft",
        # 让 [0..4] 与 [35..39] 层全部参与训练：
        extra_train_layers=[0,1,2,3,4,35,36,37,38,39],
        # 如果只想放开注意力和 MLP，不动层归一化/嵌入：
        # extra_patterns=["self_attn","mlp"],
        )
    
    
if __name__ == "__main__":
    main()