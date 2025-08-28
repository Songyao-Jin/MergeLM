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
from torch.amp import autocast, GradScaler

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



def _to_causal_lm_example(tokenizer, prompt: str, target: str, max_len: int):
    """
    拼接 prompt+target，构造 causal LM 训练所需的 input_ids / labels / attention_mask
    - prompt 部分 label = -100（不计 loss）
    - target 部分 label = token_id
    - 若末尾无 EOS：在不过长时追加；已达上限时用 EOS 覆盖最后一个 token
    """
    text = prompt + target
    
    # 先整体 tokenize（可能已被截断至 max_len）
    enc_all = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    
    # === 补/替换 EOS（检查）===
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        ids  = enc_all["input_ids"]        # [1, L]
        mask = enc_all["attention_mask"]   # [1, L]
        device, dtype = ids.device, ids.dtype
        last_id = ids[0, -1].item()

        if last_id != eos_id:
            eos = torch.tensor([[eos_id]], device=device, dtype=dtype)
            if ids.size(1) < max_len:
                # 还没到上限：直接在末尾“追加” EOS，并把 mask 也同步追加 1
                ids  = torch.cat([ids,  eos], dim=1)
                mask = torch.cat([mask, torch.ones_like(eos)], dim=1)
            else:
                # 已经被截断到上限：用 EOS 覆盖最后一个 token（长度不变）
                ids[0, -1] = eos_id
                mask[0, -1] = 1   # 显式保证 attention_mask 对应位置有效

            enc_all["input_ids"] = ids
            enc_all["attention_mask"] = mask
    
    # 构造 labels：prompt 段置为 -100，只优化 target 段
    input_ids = enc_all["input_ids"][0]
    labels    = input_ids.clone()
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    labels[:enc_prompt["input_ids"].shape[1]] = -100  
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": enc_all["attention_mask"][0],
    }
    



# ---------------- 数据集封装（Math） ----------------
class Gsm8kSFTDataset(Dataset):
    """
    openai/gsm8k (subset='main'), 监督格式：
    prompt: "Below is an instruction that describes a task.\n"
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{question}\n\n"
                "### Response:\nLet's think step by step:\n"
    target: 带推导步骤的完整答案（末尾补 EOS）
    """
    def __init__(self, tokenizer, max_len=2048, n=None, subset="main"):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []

        ds = load_dataset("openai/gsm8k", subset, split="train")
        if n:
            ds = ds.select(range(min(n, len(ds))))

        for ex in ds:
            question = (ex["question"] or "").strip()
            raw = (ex["answer"] or "").strip()
            solution, answer = raw.rsplit('####',1)
            solution, answer = solution.strip(), answer.strip()
            
            # Prompt（不计损失）
            prompt = (
                "Below is an instruction that describes a task.\n"
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{question}\n\n"
                "### Response:\nLet's think step by step:\n"
            )
            # Target（计损失）
            target = f"{solution}\n\nThe answer is: {answer}"
            
            self.samples.append((prompt, target))

        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        return _to_causal_lm_example(self.tok, prompt, target, self.max_len)


class HendrycksMathSFTDataset(Dataset):
    """
    nlile/hendrycks-MATH-benchmark，监督格式与 GSM8K 保持一致
    """
    def __init__(self, tokenizer, max_len=2048, n=None):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []

        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
        if n: 
            ds = ds.select(range(min(n, len(ds))))

        for ex in ds:
            question = (ex["problem"] or "").strip()
            solution = (ex["solution"] or "").strip()
            answer = (ex["answer"] or "").strip()
            
            # Prompt（不计损失）
            prompt = (
                "Below is an instruction that describes a task.\n"
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{question}\n\n"
                "### Response:\nLet's think step by step:\n"
            )
            # Target（计损失）
            target = f"{solution}\n\nThe final answer is: {answer}"
            
            self.samples.append((prompt, target))

        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        return _to_causal_lm_example(self.tok, prompt, target, self.max_len)



# ---------------- 数据集封装（Code Alpaca） ----------------
class CodeAlpacaDataset(Dataset):
    """
    theblackcat102/evol-codealpaca-v1 只有两个字段：
    - instruction: 任务/问题文本（可能含乱码/代码段）
    - output: 期望响应（常含代码块）
    模板（SFT风格）：
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    Create a Python script for this problem:
    {instruction}
    ### Response:
    {output}
    """
    def __init__(self, tokenizer, max_len=2048, n=None):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []

        ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
        if n:
            ds = ds.select(range(min(n, len(ds))))

        for ex in ds:
            instruction = (ex.get("instruction") or "").strip()
            output   = (ex.get("output") or "").strip()
            if not instruction or not output:
                continue  # 跳过无效样本

            # Prompt（不计损失）
            prompt = (
                "Below is an instruction that describes a task.\n "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                "Create a Python script for this problem:\n"
                f"{instruction}\n\n"
                "### Response:\n"
            )

            # Target（计损失）
            target = output

            self.samples.append((prompt, target))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        # 统一用工具：补/替换 EOS，构造 labels（prompt 段置 -100）
        return _to_causal_lm_example(self.tok, prompt, target, self.max_len)



# ---------------- 数据集封装（Evol-Instruct V2） ----------------
class EvolInstructV2Dataset(Dataset):
    """
    WizardLMTeam/WizardLM_evol_instruct_V2_196k
    每条数据结构：
      - idx: id
      - conversations: [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]
    只取单轮 human -> gpt 作为监督样本
    Prompt 模板：
      A chat between a curious user and an artificial intelligence assistant.
      The assistant gives helpful, detailed, and polite answers to the user's questions.
      USER: {instruction}
      ASSISTANT:
    """
    def __init__(self, tokenizer, max_len=2048, n=None):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []

        ds = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train")
        if n:
            ds = ds.select(range(min(n, len(ds))))

        for ex in ds:
            conversations = ex.get("conversations", [])
            # 过滤空/畸形
            if not isinstance(conversations, list) or len(conversations) < 2:
                continue
            
            if conversations[0]["from"] != "human" or conversations[1]["from"] != "gpt":
                continue

            instruction = conversations[0]["value"].strip()
            response    = conversations[1]["value"].strip()
            
            prompt = (
                "A chat between a curious user and an artificial intelligence assistant.\n"
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                f"### USER:\n {instruction}\n\n"
                "### ASSISTANT:\n"
            )
            target = response
            
            self.samples.append((prompt, target))
            
        random.shuffle(self.samples)
            
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        prompt, target = self.samples[i]
        return _to_causal_lm_example(self.tok, prompt, target, self.max_len)



# === Round-Robin（耗尽式）平衡采样数据集 ===
class RoundRobinConcat(Dataset):
    """
    将多个子 Dataset 以“轮番直到耗尽”的顺序拼接：
    - 初始打乱每个子数据集的索引（可复现，用 seed）
    - 每一“轮”按 ds0, ds1, ..., dsN 依次各取 1 条
    - 若某个子集已取完，则在后续轮次中跳过该子集
    - 直到所有子集都取完为止
    - per_dataset_take: 限制每个子集最多取多少条（None 表示取完该子集）
    """
    def __init__(self, datasets: List[Dataset], per_dataset_take: Optional[int] = None, seed: int = 42):
        assert len(datasets) > 0, "need at least one dataset"
        self.datasets = datasets
        rng = random.Random(seed)

        # 为每个子集生成一个打乱后的索引列表；必要时裁到 per_dataset_take
        self._buckets: List[List[int]] = []
        for d in datasets:
            idxs = list(range(len(d)))
            rng.shuffle(idxs)
            if per_dataset_take is not None:
                idxs = idxs[:min(per_dataset_take, len(idxs))]
            self._buckets.append(idxs)

        # 构建“耗尽式轮番”的全局访问顺序
        order: List[Tuple[int, int]] = []
        positions = [0] * len(self._buckets)  # 每个子集当前游标
        num_finished = 0
        finished = [False] * len(self._buckets)

        while num_finished < len(self._buckets):
            progressed_this_round = False
            for j, bucket in enumerate(self._buckets):
                if finished[j]:
                    continue
                if positions[j] < len(bucket):
                    local_idx = bucket[positions[j]]
                    positions[j] += 1
                    order.append((j, local_idx))
                    progressed_this_round = True
                if positions[j] >= len(bucket) and not finished[j]:
                    finished[j] = True
                    num_finished += 1
            # 理论上当所有 bucket 都空时才会退出；这里加保护避免死循环
            if not progressed_this_round:
                break

        self.order = order  # [(ds_id, local_idx), ...]，直到把所有子集耗尽

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        ds_id, local_idx = self.order[idx]
        return self.datasets[ds_id][local_idx]






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
    task_mix: dict = {"gsm8k":1.0, "math":1.0, "code":1.0, "general": 1.0},
    math_sizes=(8000, 8000),
    code_n=50000,
    general_n=30000,                                              
    lr=1e-5,
    epochs=1,
    batch_size=1,
    grad_accum=16,
    wandb_proj="sparse-ft",
    run_name="sparse_ft_run",
    seed=0,
    # 可选：给全参/稀疏不同的 weight_decay
    wd_sparse: float = 0.0, wd_dense: float = 0.01,
    # ↓↓↓ 新增：想额外放开的层索引
    extra_train_layers: Optional[List[int]] = None,
    extra_patterns: Optional[List[str]] = None,  # 例如只放开 ["self_attn","mlp"]
    # 模型训练提速的参数
    use_amp: bool = True,        # ← 是否启用 AMP
    
):
    set_random_seed(seed)
    # 1) 加载模型
    print("load")
    model, tok = load_model_and_tokenizer(base_model_name_or_path, half_model_dtype=False, seed=seed, device="auto")
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
    ds_list: List[Dataset] = []
    if task_mix.get("gsm8k", 0) > 0:
        ds_list.append(Gsm8kSFTDataset(tok, max_len=1024, n=math_sizes[0]))
    if task_mix.get("math", 0) > 0:
        ds_list.append(HendrycksMathSFTDataset(tok, max_len=1024, n=math_sizes[1]))
    if task_mix.get("code", 0) > 0:
        ds_code = CodeAlpacaDataset(tok, max_len=1024, n=code_n)
        ds_list.append(ds_code)
    if task_mix.get("general", 0) > 0:
        ds_gen = EvolInstructV2Dataset(tok, max_len=1024, n=general_n)
        ds_list.append(ds_gen)
        
    
    # 严格轮番
    # 将多个子 Dataset 以“轮番直到耗尽”的顺序拼接
    # DataLoader 不需要 shuffle（顺序已在数据集里确定）
    train_ds = RoundRobinConcat(ds_list, per_dataset_take=None, seed=seed)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id)
    )
    
    
    # 5) 优化器 & 调度
    if train_all_weights:
        weight_decay = wd_dense
    else:
       weight_decay = wd_sparse
       
    # try:
    #     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    # except TypeError:
    #     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / (batch_size * grad_accum))
    num_steps = steps_per_epoch * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * num_steps), num_steps)
    
    # ===== AMP/Scaler =====
    _amp_dtype = None
    if torch.cuda.is_bf16_supported():
        _amp_dtype = torch.bfloat16
        print("Use torch.bfloat16 for fine tuning.")
    else:
        _amp_dtype = torch.float16
        print("Use torch.float16 for fine tuning.")
    scaler = GradScaler('cuda', enabled=(use_amp and torch.cuda.is_available() and _amp_dtype == torch.float16))    #AMP/Scaler
    
    # ====== Save plan & helper ======
    # 保存计划：前 8 次每 128 step，后 4 次每 256 step
    first_k, first_gap = 8, 128
    second_k, second_gap = 4, 256
    
    
    first_part  = [first_gap * i for i in range(1, first_k + 1)]
    start_after = first_part[-1] if first_part else 0
    second_part = [start_after + second_gap * i for i in range(1, second_k + 1)]

    save_steps = set(first_part + second_part)   # 用 set 方便快速查
    
    
    # 6) W&B
    if wandb_proj:
        wandb.init(project=wandb_proj, name=run_name, config={
            "base": base_model_name_or_path,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "grad_accum": grad_accum,
            "math_sizes": math_sizes, "code_n": code_n, "general_n": general_n,
        })
        
    # 7) 训练
    update_step = 0
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            with autocast("cuda", enabled=use_amp, dtype=_amp_dtype):
                out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                loss = out.loss / grad_accum
            
            
            # 反传：fp16 用 scaler.scale。bf16/关闭AMP，则普通 backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            running += loss.item()
            
            if (i + 1) % grad_accum == 0:
                if train_all_weights == False:
                    # 在 step 前做一次“步前梯度掩码”，避免 mask 常驻 GPU. (注意在 unscale 前也可以，因为只是逐元素乘)
                    apply_grad_masks_step_(model, merged_mask)  # 稀疏才需要
                
                # 如果用了 scaler，需要先 unscale 再做 clip
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 优化器步进
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                
                opt.zero_grad()
                sch.step()
                
                update_step += 1
                
                # 触发：在计划中的 step 才保存
                if update_step in save_steps:
                    ckpt_dir = os.path.join(out_dir, f"epoch{epoch}_batchSize{batch_size}_gradAccum{grad_accum}_ckpt-step{update_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tok.save_pretrained(ckpt_dir)
                    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                        json.dump({
                            "epoch": epoch,
                            "update_step": update_step,
                            "batch_size": batch_size,
                            "grad_accum": grad_accum,
                            "lr": lr
                        }, f, ensure_ascii=False, indent=2)
                    print(f"[CKPT] Saved checkpoint at step {update_step} -> {ckpt_dir}")
                
                if wandb_proj and update_step % 10 == 0:
                    wandb.log({"loss": running, "lr": sch.get_last_lr()[0], "step": update_step})
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
    
    # sparse finetune
    sparse_finetune(
        base_model_name_or_path="Llama-2-13b-hf",
        merged_sparse_delta_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_sparse_delta_union_avg_top10.pt",
        merged_mask_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_mask_union_top10.pt",
        out_dir="sparse_ft_ckpts/llama2-13b-math-code-alignment-sparseft",
        task_mix={"gsm8k":1.0, "math":1.0, "code":1.0, "general": 1.0},   # 做数学+代码+对齐
        math_sizes=(8000, 8000),             # GSM8K 8k + MATH 8k（可按显存调小）
        code_n=8000,                        # CodeAlpaca 子集
        general_n=8000,
        lr=1e-5, epochs=1, batch_size=1, grad_accum=16,
        wandb_proj="weight-sparse-ft", run_name="llama2-13b-sparseft",
        use_amp = True, 
        )
    
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
    
    
    # # sparse finetune with extra_train_layers
    # sparse_finetune(
    #     base_model_name_or_path="Llama-2-13b-hf",
    #     merged_sparse_delta_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_sparse_delta_union_avg_top10.pt",
    #     merged_mask_path="sparse_merge_outputs/Llama-2-13b-hf_f53da205/merged_mask_union_top10.pt",
    #     out_dir="sparse_ft_ckpts/llama2-13b-math-code-alignment-estra_train_layers_[0,1,2,3,4,35,36,37,38,39]-sparseft",
    #     task_mix={"math":1.0, "code":1.0, "general": 1.0},   # 做数学+代码+对齐
    #     math_sizes=(8000, 8000),             # GSM8K 8k + MATH 8k（可按显存调小）
    #     code_n=8000,                        # CodeAlpaca 子集
    #     general_n=8000,         # 约 25 steps（示意）
    #     lr=1e-5, epochs=1, batch_size=1, grad_accum=16,
    #     wandb_proj="weight-sparse-ft", run_name="llama2-13b-sparseft",
    #     # 让 [0..4] 与 [35..39] 层全部参与训练：
    #     extra_train_layers=[0,1,2,3,4,35,36,37,38,39],
    #     # 如果只想放开注意力和 MLP，不动层归一化/嵌入：
    #     # extra_patterns=["self_attn","mlp"],
    #     )
    
    
if __name__ == "__main__":
    main()