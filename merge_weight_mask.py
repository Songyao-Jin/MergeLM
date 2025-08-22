import os
import json
from typing import Dict, List, Tuple
import torch
import hashlib 
from record_activations import load_model_and_tokenizer
from extract_key_weights import _sanitize


# ---------- 基础工具 ----------

# def load_state_dict_flex(path_or_sd):
#     """既支持 dict，也支持 torch.save/torch.load 的路径。"""
#     if isinstance(path_or_sd, dict):
#         sd = path_or_sd
#     else:
#         sd = torch.load(path_or_sd, map_location="cpu")
#         if hasattr(sd, "state_dict"):
#             sd = sd.state_dict()
#     # 统一到 CPU、float32，便于后续计算
#     sd = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
#     return sd

def load_mask_dict(path_or_dict):
    """加载 mask：{name: tensor(0/1)}，转换为 bool。"""
    md = path_or_dict if isinstance(path_or_dict, dict) else torch.load(path_or_dict, map_location="cpu")
    out = {}
    for k, v in md.items():
        if not torch.is_tensor(v):
            continue
        # 兼容 bfloat16/float 的 0/1；转 bool
        out[k] = (v != 0).to(torch.bool).cpu()
    return out

def compute_delta_dict(base_sd: Dict[str, torch.Tensor],
                       ft_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """δ = ft - base，仅对两边都有且 shape 一致且为浮点的参数。"""
    out = {}
    for k, v in ft_sd.items():
        if k in base_sd and torch.is_tensor(v) and torch.is_tensor(base_sd[k]) and v.shape == base_sd[k].shape:
            if torch.is_floating_point(v) and torch.is_floating_point(base_sd[k]):
                out[k] = (v - base_sd[k]).cpu()
    return out

# ---------- 统计重叠 ----------

def overlap_stats(masks_by_name: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
    """
    统计各模型 mask 的重叠：
      - pairwise: 对每一对 (i,j) 输出 overlap/union/jaccard/over_i/over_j
      - all_models: 全体交/并 与 Jaccard
    返回结构是可 JSON 序列化的统计字典。
    """
    names = list(masks_by_name.keys())
    # 全局计数容器
    selected_count = {n: 0 for n in names}
    # 预先计算每个模型被选总数
    for n in names:
        s = 0
        for k, m in masks_by_name[n].items():
            s += int(m.sum().item())
        selected_count[n] = s

    # pairwise
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            ni, nj = names[i], names[j]
            inter = 0
            uni = 0
            for k in set(masks_by_name[ni].keys()) | set(masks_by_name[nj].keys()):
                mi = masks_by_name[ni].get(k, None)
                mj = masks_by_name[nj].get(k, None)
                if mi is None and mj is None:
                    continue
                if mi is None:
                    mi = torch.zeros_like(mj, dtype=torch.bool)
                if mj is None:
                    mj = torch.zeros_like(mi, dtype=torch.bool)
                inter += int((mi & mj).sum().item())
                uni   += int((mi | mj).sum().item())
            over_i = inter / max(1, selected_count[ni])
            over_j = inter / max(1, selected_count[nj])
            jacc   = inter / max(1, uni)
            pairs.append({
                "model_i": ni,
                "model_j": nj,
                "overlap": inter,
                "union": uni,
                "jaccard": jacc,
                "overlap_over_i": over_i,
                "overlap_over_j": over_j,
                "selected_i": selected_count[ni],
                "selected_j": selected_count[nj],
            })

    # 全体交并
    # 先收集所有 key
    all_keys = set()
    for n in names:
        all_keys |= set(masks_by_name[n].keys())

    inter_all = 0
    union_all = 0
    for k in all_keys:
        acc_or = None
        acc_and = None
        for n in names:
            m = masks_by_name[n].get(k, None)
            if m is None:
                # 若缺失该参数，视作全 False 的同形状（但我们不知道形状），
                # 因此这里保守：如果某模型缺失该 key，无法做同形状与或。
                # 解决：只统计那些所有模型都有该 key 的交集部分。
                acc_and = None
                acc_or  = None
                break
            if acc_or is None:
                acc_or = m.clone()
                acc_and = m.clone()
            else:
                acc_or  |= m
                acc_and &= m
        if acc_or is None:  # 有模型缺这个 key，跳过
            continue
        union_all += int(acc_or.sum().item())
        inter_all += int(acc_and.sum().item())

    stats = {
        "selected_per_model": selected_count,
        "pairwise": pairs,
        "all_models": {
            "intersection": inter_all,
            "union": union_all,
            "jaccard": (inter_all / max(1, union_all)),
        }
    }
    return stats

# ---------- 合并 mask 并聚合稀疏 delta ----------

def merge_masks(masks_list: List[Dict[str, torch.Tensor]],
                mode: str = "union") -> Dict[str, torch.Tensor]:
    """
    合并多个 mask：返回 merged_mask_dict（每 key 一个 bool mask）
      mode: "union" | "intersection"
    """
    assert mode in ("union", "intersection")
    out = {}
    all_keys = set()
    for md in masks_list:
        all_keys |= set(md.keys())
    for k in all_keys:
        merged = None
        for md in masks_list:
            m = md.get(k, None)
            if m is None:
                continue
            if merged is None:
                merged = m.clone().bool()
            else:
                merged = (merged | m) if mode == "union" else (merged & m)
        if merged is None:
            continue
        out[k] = merged.bool()
    return out

def aggregate_sparse_delta(base_sd: Dict[str, torch.Tensor],
                           delta_list: List[Dict[str, torch.Tensor]],
                           masks_list: List[Dict[str, torch.Tensor]],
                           merge_mask: Dict[str, torch.Tensor],
                           agg: str = "avg") -> Dict[str, torch.Tensor]:
    """
    先对齐每个参数，每个位置上只取“被各自模型选中”的 delta 参与聚合（求和/平均），然后再用 merge_mask 把输出严格裁成想要的稀疏形状。
    
    根据各模型的 delta 与其 mask，按 agg 聚合出 merged 稀疏 delta（只在 merge_mask=1 处保留）。
      agg: "avg" | "sum" （也可扩展 "max_abs"）
    """
    assert agg in ("avg", "sum")
    N = len(delta_list)
    out = {}
    # 遍历所有可能的 key
    keys = set(merge_mask.keys())
    # for d in delta_list:
    #     keys |= set(d.keys())
    for k in keys:
        # if k not in merge_mask:
        #     continue
        mm = merge_mask[k]  # bool
        if k not in base_sd:
            # 没有 base 对齐就跳过
            continue

        # 堆叠各模型的 delta / mask（缺失视作 0 / False）
        deltas = []
        masks  = []
        shape = base_sd[k].shape
        for i in range(N):
            di = delta_list[i].get(k, None)
            mi = masks_list[i].get(k, None)
            if di is None:
                di = torch.zeros(shape, dtype=torch.bfloat16)
            if mi is None:
                mi = torch.zeros(shape, dtype=torch.bool)
            deltas.append(di)
            masks.append(mi)
        D = torch.stack(deltas, dim=0)  # [N, *]
        M = torch.stack(masks,  dim=0)  # [N, *] bool

        # 只对各自选中的位置参与聚合
        Dm = D * M.to(D.dtype)          # [N, *] # 把没选中的位置的 delta 置零。  直觉：只在各自 mask=True 的位置参与聚合；未选中的模型不会“稀释”平均值。
        if agg == "sum":
            merged = Dm.sum(dim=0)
        elif agg == "avg":
            denom = M.sum(dim=0)        # 每个位置被多少个模型选中.  
            denom = denom.clamp_min(1)  # 防止除0
            merged = Dm.sum(dim=0) / denom

        # 只在 merge_mask=1 处保留（其他位置置 0）
        merged = merged * mm.to(merged.dtype)
        out[k] = merged.cpu()
    return out

# ---------- 顶层封装：一键生成 merged 稀疏 delta + 重叠统计 ----------

def build_auto_paths(
    base_name: str,
    ft_names: List[str],
    merge_mode: str,
    agg: str,
    root_dir: str = "sparse_merge_outputs",
    tag: str = ""
) -> Tuple[str, str, str]:
    """
    返回三个路径:
      out_merged_delta_path, out_overlap_json, out_merged_mask_path
    会自动生成子目录: {root}/{base}__{hash_of_ft_names}/
    """
    b = _sanitize(base_name)
    ft_join = "++".join(_sanitize(n) for n in ft_names)
    # ft_hash = hashlib.md5(ft_join.encode()).hexdigest()[:8]  # 防超长
    ft_hash = ft_join
    subdir = os.path.join(root_dir, f"{b}_{ft_hash}")
    os.makedirs(subdir, exist_ok=True)
    tag2 = f"_{tag}" if tag else ""
    merged_delta = os.path.join(subdir, f"merged_sparse_delta_{merge_mode}_{agg}{tag2}.pt")
    overlap_json = os.path.join(subdir, f"mask_overlap_{merge_mode}{tag2}.json")
    merged_mask  = os.path.join(subdir, f"merged_mask_{merge_mode}{tag2}.pt")
    return merged_delta, overlap_json, merged_mask

def state_dict_from_model_name(model_name: str) -> Dict[str, torch.Tensor]:
    """
    用你的 load_model_and_tokenizer 加载模型，然后取 state_dict，并统一到 CPU/float32。
    """
    model, _ = load_model_and_tokenizer(model_name, half_model_dtype=False, seed=0)
    sd = model.state_dict()
    # 统一 detatch+cpu+float，避免设备不一致
    sd = {k: (v.detach().cpu().float() if torch.is_tensor(v) else v) for k, v in sd.items()}
    # 清理显存（如果用了 GPU）
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    return sd


def build_sparse_merged_delta(
    base_model_name: str=None,
    ft_model_names: List[str]=None,

    masks_paths: List[str]=None,          # 与 finetuned 一一对应
    merge_mode: str = "union",            # "union" | "intersection"
    agg: str = "avg",                     # "avg" | "sum"

    # 输出：若不指定则自动生成
    out_merged_delta_path: str=None,
    out_overlap_json: str=None,
    save_merged_mask: bool=True,
    out_merged_mask_path: str=None,

    # 自动命名所需信息
    auto_root_dir: str = "sparse_merge_outputs",
    auto_tag: str = "",   # 你可以传 "top10" 或数据集名等
) -> Tuple[Dict, Dict, Dict]:
    """
    返回 (merged_sparse_delta_dict, overlap_stats_dict, merged_mask_dict)
    同时把 merged_sparse_delta / overlap_json / merged_mask(可选) 存盘。
    """
    assert masks_paths is not None and len(masks_paths) > 0, "必须提供 masks_paths"
    # 1) 解析 base 与 ft 的 state_dict
    assert base_model_name is not None, "需提供 base_sd_or_path 或 base_model_name"
    base_sd = state_dict_from_model_name(base_model_name)


    assert ft_model_names is not None, "需提供 ft_sds_or_paths 或 ft_model_names"
    ft_names = ft_model_names
    ft_sd_list = [state_dict_from_model_name(n) for n in ft_model_names]
    

    assert len(ft_sd_list) == len(masks_paths), "finetuned 数量要与 masks 一一对应"

    # 2) 如果没给输出路径，自动生成
    if out_merged_delta_path is None or out_overlap_json is None or (save_merged_mask and out_merged_mask_path is None):
        base_name_for_path = base_model_name if base_model_name is not None else "BASE"
        ft_names_for_path  = ft_names
        m_delta, m_overlap, m_mask = build_auto_paths(base_name_for_path, ft_names_for_path, merge_mode, agg, root_dir=auto_root_dir, tag=auto_tag)
        out_merged_delta_path = out_merged_delta_path or m_delta
        out_overlap_json      = out_overlap_json or m_overlap
        out_merged_mask_path  = out_merged_mask_path or m_mask

    # 3) 计算每个 finetuned 的 delta 与读取 mask
    delta_list = []
    masks_by_name = {}
    masks_list = []
    for idx, (ft_sd, mpath) in enumerate(zip(ft_sd_list, masks_paths)):
        delta_i = compute_delta_dict(base_sd, ft_sd)   # δ_i = ft - base
        mask_i  = load_mask_dict(mpath)                # {name: bool tensor}
        delta_list.append(delta_i)

        name = ft_names[idx] if idx < len(ft_names) else f"model_{idx}"
        masks_by_name[name] = mask_i
        masks_list.append(mask_i)

    # 4) 合并 mask + 重叠统计
    merged_mask = merge_masks(masks_list, mode=merge_mode)   # bool dict
    stats = overlap_stats(masks_by_name)                     # JSON-able dict

    # 5) 聚合得到 merged 稀疏 delta（只保留 merged_mask=1 的位置）
    merged_sparse_delta = aggregate_sparse_delta(
        base_sd, delta_list, masks_list, merged_mask, agg=agg
    )

    # 6) 保存
    os.makedirs(os.path.dirname(out_merged_delta_path) or ".", exist_ok=True)
    torch.save(merged_sparse_delta, out_merged_delta_path)

    os.makedirs(os.path.dirname(out_overlap_json) or ".", exist_ok=True)
    with open(out_overlap_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if save_merged_mask:
        os.makedirs(os.path.dirname(out_merged_mask_path) or ".", exist_ok=True)
        torch.save(merged_mask, out_merged_mask_path)

    print(f"[OK] merged 稀疏 delta 已保存: {out_merged_delta_path}")
    print(f"[OK] mask 重叠统计已保存: {out_overlap_json}")
    if save_merged_mask:
        print(f"[OK] merged mask 已保存: {out_merged_mask_path}")

    return merged_sparse_delta, stats, merged_mask


def main():
    base_name = "Llama-2-13b-hf"
    ft_names  = ["WizardLM-13B-V1.2",
                "llama-2-13b-code-alpaca",
                "WizardMath-13B-V1.0"]
    mask_paths = ["masks/Llama-2-13b-hf__WizardLM-13B-V1.2/top10_to_top0__bias-norm-embed__bf16.pt",
                    "masks/Llama-2-13b-hf__llama-2-13b-code-alpaca/top10_to_top0__bias-norm-embed__bf16.pt", 
                    "masks/Llama-2-13b-hf__WizardMath-13B-V1.0/top10_to_top0__bias-norm-embed__bf16.pt"]

    merged_delta, stats, merged_mask = build_sparse_merged_delta(
        base_model_name=base_name,
        ft_model_names=ft_names,
        masks_paths=mask_paths,
        merge_mode="union",       # 或 "intersection"
        agg="avg",                # 或 "sum"
        auto_root_dir="sparse_merge_outputs",
        auto_tag="top10",         # 你想写的任何标记
        save_merged_mask=True,
    )




if __name__ == "__main__":
    main()











# def build_sparse_merged_delta(
#     base_sd_or_path,
#     ft_sds_or_paths: List,             # 与 masks_paths 一一对应
#     masks_paths: List[str],             # 每个 finetuned 的 mask 路径
#     merge_mode: str = "union",          # "union" | "intersection"
#     agg: str = "avg",                   # "avg" | "sum"
#     out_merged_delta_path: str = "merged_sparse_delta.pt",
#     out_overlap_json: str = "mask_overlap_stats.json",
# ) -> Tuple[Dict, Dict]:
#     """
#     返回 (merged_sparse_delta_dict, overlap_stats_dict)，并保存到文件。
#     """
#     assert len(ft_sds_or_paths) == len(masks_paths), "ft_sds_or_paths 与 masks_paths 数量需一致"

#     base_sd = load_state_dict_flex(base_sd_or_path)

#     # 每个 finetuned 的 delta + mask
#     delta_list = []
#     masks_by_name = {}
#     masks_list = []

#     for idx, (ft, mpath) in enumerate(zip(ft_sds_or_paths, masks_paths)):
#         ft_sd   = load_state_dict_flex(ft)
#         delta_i = compute_delta_dict(base_sd, ft_sd)
#         mask_i  = load_mask_dict(mpath)
#         delta_list.append(delta_i)

#         name = f"model_{idx}"
#         masks_by_name[name] = mask_i
#         masks_list.append(mask_i)

#     # 合并 mask（用于最后导出的 merged 稀疏 delta 的“选中范围”）
#     merged_mask = merge_masks(masks_list, mode=merge_mode)

#     # 统计重叠
#     stats = overlap_stats(masks_by_name)

#     # 聚合得到 merged 稀疏 delta
#     merged_sparse_delta = aggregate_sparse_delta(
#         base_sd, delta_list, masks_list, merged_mask, agg=agg
#     )

#     # 保存
#     os.makedirs(os.path.dirname(out_merged_delta_path) or ".", exist_ok=True)
#     torch.save(merged_sparse_delta, out_merged_delta_path)
#     os.makedirs(os.path.dirname(out_overlap_json) or ".", exist_ok=True)
#     with open(out_overlap_json, "w", encoding="utf-8") as f:
#         json.dump(stats, f, ensure_ascii=False, indent=2)

#     print(f"[OK] merged 稀疏 delta 已保存: {out_merged_delta_path}")
#     print(f"[OK] mask 重叠统计已保存: {out_overlap_json}")
#     return merged_sparse_delta, stats


