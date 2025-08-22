import os
# 🔐 Redirect all Hugging Face-related caches and locks to your personal directory
os.environ["HF_HOME"] = "/data/songyao/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/songyao/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/songyao/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/data/songyao/.cache/huggingface/hub"



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.load_config import cache_dir
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
import numpy as np
from record_activations import load_model_and_tokenizer








def calc_delta_weights_from_models(finetune_model, base_model, verbose=True):
    """
    计算 finetune_model 与 base_model 的参数差（delta），以 state_dict 形式返回。
    要求两个模型完全 shape 对齐。
    返回: delta_state_dict
    """
    finetune_sd = finetune_model.state_dict()
    base_sd = base_model.state_dict()
    delta_sd = {}

    all_keys = set(finetune_sd.keys()) & set(base_sd.keys())
    extra_keys_finetune = set(finetune_sd.keys()) - set(base_sd.keys())
    extra_keys_base = set(base_sd.keys()) - set(finetune_sd.keys())

    if verbose:
        if extra_keys_finetune:
            print(f"[警告] finetune 模型比 base 多出参数: {extra_keys_finetune}")
        if extra_keys_base:
            print(f"[警告] base 模型比 finetune 多出参数: {extra_keys_base}")

    for k in all_keys:
        v_finetune = finetune_sd[k]
        v_base = base_sd[k]
        if v_finetune.shape != v_base.shape:
            raise ValueError(f"参数 {k} 维度不一致: finetune {v_finetune.shape}, base {v_base.shape}")
        delta = v_finetune.detach() - v_base.detach()
        delta_sd[k] = delta
        # 可选：对纯 embedding int 参数等不做delta
        # if v_finetune.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        #     delta_sd[k] = v_finetune.detach().clone()

    return delta_sd



def load_wanda_metrics(npz_path, verbose=True):
    """
    加载 wanda 指标 npz 文件，返回 {module_name: wanda_matrix} 字典
    参数:
        npz_path: wanda 指标保存路径（如 .../wizardmath_wanda_metrics.npz）
    返回:
        wanda_dict: {str: np.ndarray}
    """
    wanda_data = np.load(npz_path, allow_pickle=True)
    wanda_dict = {k: wanda_data[k] for k in wanda_data.files}
    if verbose:
        print(f"Loaded wanda metrics from {npz_path}: {list(wanda_dict.keys())}")
    return wanda_dict



def merge_delta_weight_wise_normalized(
    delta1, delta2, wanda1, wanda2, merge_strategy="softmax", eps=1e-6, amplification_factor = 1
):
    """
    用归一化相对重要性 (4*(w1-w2)/(w1+w2+eps)) 作为 sigmoid 输入
    """
    if not isinstance(delta1, torch.Tensor):
        delta1 = torch.from_numpy(delta1)
    if not isinstance(delta2, torch.Tensor):
        delta2 = torch.from_numpy(delta2)
    if not isinstance(wanda1, torch.Tensor):
        wanda1 = torch.from_numpy(wanda1)
    if not isinstance(wanda2, torch.Tensor):
        wanda2 = torch.from_numpy(wanda2)
    assert delta1.shape == delta2.shape == wanda1.shape == wanda2.shape

    # s = 4.0 * (wanda1 - wanda2) / (wanda1 + wanda2 + eps)
    s = 4.0 * (wanda1 - wanda2)
    # 其他还有 “欧氏距离归一化”方式
    if merge_strategy == "hard":
        mask = wanda1 > wanda2
        alpha = mask.float()
    elif merge_strategy == "softmax":
        alpha = torch.sigmoid(s)
    merged_delta = amplification_factor*(alpha * delta1 + (1.0 - alpha) * delta2)
    return merged_delta



def merge_arithmetic_delta(delta1, delta2, alpha=0.5):
    """
    对一维参数 delta 做 task arithmetic 融合
    delta1, delta2: torch.Tensor 或 np.ndarray，形状必须相同且为1维
    alpha: float，融合系数，alpha=1只用delta1，0只用delta2，0.5等权平均
    返回: merged_delta
    """
    if not isinstance(delta1, torch.Tensor):
        delta1 = torch.from_numpy(delta1)
    if not isinstance(delta2, torch.Tensor):
        delta2 = torch.from_numpy(delta2)
    assert delta1.shape == delta2.shape
    # assert delta1.ndim == 1, "只能用于一维参数"
    # merged_delta = alpha * delta1 + (1.0 - alpha) * delta2
    merged_delta = alpha * delta1 + alpha * delta2
    return merged_delta





def assemble_merged_model(base_model, merged_delta_dict, verbose=True):
    """
    输入:
        base_model: transformers加载的模型 或 base_model.state_dict()
        merged_delta_dict: {参数名: delta张量}，与base_model参数同名同shape
    输出:
        merged_state_dict: 新权重dict，可直接用于 save/load
    """
    # 支持 base_model 为模型对象或 state_dict
    if hasattr(base_model, "state_dict"):
        base_sd = base_model.state_dict()
    else:
        base_sd = base_model  # 已是 state_dict

    merged_state_dict = {}
    mismatch_keys = []
    for k in base_sd:
        if k in merged_delta_dict:
            try:
                merged_state_dict[k] = base_sd[k] + merged_delta_dict[k].to(base_sd[k].device, dtype=base_sd[k].dtype)
            except Exception as e:
                mismatch_keys.append((k, base_sd[k].shape, merged_delta_dict[k].shape))
                if verbose:
                    print(f"警告: 参数 {k} shape 不一致, base {base_sd[k].shape}, delta {merged_delta_dict[k].shape}")
                # 直接用base参数
                merged_state_dict[k] = base_sd[k]
        else:
            merged_state_dict[k] = base_sd[k]  # 没有delta的参数用base
            print(f"警告: 在delta_dict中未找到参数{k}")

    if verbose and mismatch_keys:
        print(f"[警告] 以下参数shape不匹配, 已用base参数: {mismatch_keys}")

    return merged_state_dict






def save_state_dict(state_dict, out_path, verbose=True):
    """
    保存 PyTorch state_dict 到本地文件.
    参数:
        state_dict: dict，模型参数
        out_path: 保存文件名 (如 'merged_model.pth')
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict, out_path)
    if verbose:
        print(f"已保存新权重到: {out_path}")


def save_hf_pretrained(base_model, base_tokenizer, merged_state_dict, hf_save_dir):
    base_model.load_state_dict(merged_state_dict)
    base_model.save_pretrained(hf_save_dir)
    base_tokenizer.save_pretrained(hf_save_dir)
    print(f"融合模型已保存为 HuggingFace 目录: {hf_save_dir}")



def find_wanda_key(k, wanda_dict):
    # 优先精确查找
    if k in wanda_dict:
        return k
    # 如果k是xx.weight结尾，而wanda_dict是没.weight的，strip后查找
    if k.endswith('.weight'):
        k_no_weight = k[:-7]
        if k_no_weight in wanda_dict:
            return k_no_weight
    return None

def run_model_merging_with_wanda(
    base_model_name,
    math_model_name,
    code_model_name,
    wanda_math_path,
    wanda_code_path,
    out_path,
    merge_strategy="softmax",    # "softmax" or "hard"
    norm_param_names=("norm", "layernorm", "ln"),  # 用于一维参数判断
    alpha_norm=0.5,           # norm层融合系数
    amplification_factor = 1,
    verbose=True,
):
    """
    用于精细化融合 WizardMath-13B (Math) 和 llama-2-13b-codealpaca (Code) 两模型
    """
    # ===== 1. 加载模型 =====
    print("加载 base, math, code 三个模型 ...")
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, half_model_dtype=False, seed=0)
    math_model, _ = load_model_and_tokenizer(math_model_name, half_model_dtype=False, seed=0)
    code_model, _ = load_model_and_tokenizer(code_model_name, half_model_dtype=False, seed=0)

    # ===== 2. 计算 delta =====
    print("计算 delta weights ...")
    delta_math = calc_delta_weights_from_models(math_model, base_model)
    delta_code = calc_delta_weights_from_models(code_model, base_model)

    # ===== 3. 加载 wanda 指标 =====
    print("加载 wanda 指标 ...")
    wanda_math = load_wanda_metrics(wanda_math_path)
    wanda_code = load_wanda_metrics(wanda_code_path)

    # ===== 4. 融合 delta =====
    print("开始融合 delta ...")
    merged_delta = {}
    for k in delta_math:
        v1, v2 = delta_math[k], delta_code[k]
        k_wanda1 = find_wanda_key(k, wanda_math)
        k_wanda2 = find_wanda_key(k, wanda_code)
        
        # 线性/MLP参数（权重）用 weight-wise wanda 融合
        if (k_wanda1 is not None) and (k_wanda2 is not None) and (v1.ndim == 2):
            assert v1.shape == v2.shape == wanda_math[k_wanda1].shape == wanda_code[k_wanda2].shape
            merged_delta[k] = merge_delta_weight_wise_normalized(
                v1, v2, wanda_math[k_wanda1], wanda_code[k_wanda2], merge_strategy=merge_strategy, amplification_factor = amplification_factor
            )
        # 一维参数（norm, bias 等）
        elif any(sub in k.lower() for sub in norm_param_names) and v1.ndim == 1:
            merged_delta[k] = merge_arithmetic_delta(v1, v2, alpha=alpha_norm)
        # 3. embed_tokens.weight 特判
        elif k.endswith("embed_tokens.weight") and v1.ndim == 2:
            merged_delta[k] = merge_arithmetic_delta(v1, v2, alpha=0.5)  # 等权平均
        else:
            # fallback: 均值
            merged_delta[k] = (v1 + v2) / 2
            if verbose:
                print(f"注意: {k} 未用 wanda 融合, 直接均值.")

    # ===== 5. 合成新模型权重 =====
    print("合成新模型权重 ...")
    merged_state_dict = assemble_merged_model(base_model, merged_delta)

    # ===== 6. 保存 =====
    save_state_dict(merged_state_dict, out_path)
    print(f"完成！融合模型已保存到 {out_path}")
    
    hf_save_dir = out_path.replace('.pth', '_hf')
    save_hf_pretrained(base_model, base_tokenizer, merged_state_dict, hf_save_dir)


if __name__ == "__main__":
    run_model_merging_with_wanda(
        base_model_name="Llama-2-13b-hf",
        math_model_name="WizardMath-13B-V1.0",
        code_model_name="llama-2-13b-code-alpaca",
        wanda_math_path="activations/gsm8k/n64/WizardMath-13B-V1.0_wanda_metrics.npz",
        wanda_code_path="activations/humaneval/n64/llama-2-13b-code-alpaca_wanda_metrics.npz",
        out_path="merged_models/llama-2-13b-math-code-wanda-merge_softmax_amplification_factor_2_alpha_norm_1.pth",
        merge_strategy="softmax",      # 或 "hard"
        alpha_norm=1,             # norm/bias 等一维参数用等权
        amplification_factor = 2,
        verbose=True,
    )

