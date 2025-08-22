# check_mask_entry.py
import torch

def load_mask_dict(path: str):
    m = torch.load(path, map_location="cpu")
    if not isinstance(m, dict):
        raise TypeError(f"{path} 不是字典（mask 应为 name->tensor 的 dict）")
    return m

def get_mask_value(mask_dict, name: str, idx):
    """返回 (存在该参数?, 取值tensor.item(), 是否被选中bool)"""
    if name not in mask_dict:
        return False, None, False
    t = mask_dict[name]
    if not torch.is_tensor(t):
        raise TypeError(f"mask[{name}] 不是 tensor")
    # 尽量转成 CPU
    t = t.to("cpu")
    # 越界检查
    if len(idx) != t.ndim:
        raise ValueError(f"索引维度不匹配: 索引{len(idx)}维, 但张量是{t.ndim}维, shape={tuple(t.shape)}")
    for i, s in zip(idx, t.shape):
        if i < 0 or i >= s:
            raise IndexError(f"索引 {idx} 越界，张量 shape={tuple(t.shape)}")
    v = t[idx].item()
    # 规范化为是否“选中”
    if t.dtype == torch.bool:
        selected = bool(v)
    else:
        selected = float(v) > 0.5
    return True, v, selected

if __name__ == "__main__":
    # ===== 把这三个路径替换为你的三个模型的 mask 路径 =====
    mask_paths = {
        "WizardMath-13B-V1.0":    "masks/Llama-2-13b-hf__WizardMath-13B-V1.0/top10_to_top0__bias-norm-embed__bf16.pt",
        "CodeAlpaca-13B":         "masks/Llama-2-13b-hf__llama-2-13b-code-alpaca/top10_to_top0__bias-norm-embed__bf16.pt",
        "WizardLM-13B-V1.2":      "masks/Llama-2-13b-hf__WizardLM-13B-V1.2/top10_to_top0__bias-norm-embed__bf16.pt",
    }

    # 要检查的参数名与索引（二维权重的 [out_channel, in_channel]）
    param_name = "model.layers.3.mlp.down_proj.weight"
    idx = (4743, 7678)

    print(f"检查: {param_name}{idx}\n")
    presence = {}
    for name, p in mask_paths.items():
        try:
            md = load_mask_dict(p)
            ok, val, sel = get_mask_value(md, param_name, idx)
            if not ok:
                print(f"[{name}] ⚠️ mask 中没有键: {param_name}")
                presence[name] = {"has_key": False, "value": None, "selected": False}
            else:
                dtype = md[param_name].dtype
                print(f"[{name}] 有该键 | 值={val} (dtype={dtype}) | 选中? {sel}")
                presence[name] = {"has_key": True, "value": val, "selected": sel}
        except Exception as e:
            print(f"[{name}] 读取失败: {e}")
            presence[name] = {"has_key": False, "value": None, "selected": False}

    # 汇总
    n_selected = sum(1 for v in presence.values() if v["selected"])
    n_has_key  = sum(1 for v in presence.values() if v["has_key"])
    print(f"\n汇总: 有键 {n_has_key}/{len(mask_paths)}，被选中 {n_selected}/{len(mask_paths)}")
