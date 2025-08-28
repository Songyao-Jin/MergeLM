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
import torch.nn as nn
import numpy as np
import os
import gc
from datasets import load_dataset


def load_model_and_tokenizer(model_name, half_model_dtype=False, seed=0, device ="cpu"):
    """
    加载指定 HuggingFace 格式的 LLM 模型和分词器，并自动放到 GPU。
    支持 fp16，适配大部分 LLaMA/CodeAlpaca/WizardMath 等模型。

    参数:
        model_name: str, HuggingFace hub 上的模型名或本地路径

    返回:
        model: 加载好的模型（已放到 device）
        tokenizer: 加载好的分词器
    """
    print(f"正在加载模型: {model_name}")
    
    # 1. 加载模型和 tokenizer，支持本地和远程
    if cache_dir is not None and os.path.exists(os.path.join(cache_dir, model_name)):
        model_path = os.path.join(cache_dir, model_name)
    else:
        model_path = model_name
        
    max_mem = {i: "15GiB" for i in range(torch.cuda.device_count())}  # 给每卡留 ~1GiB 余量
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, device_map=device, max_memory=max_mem, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False)
    model.eval()
    
    # model.to("cpu")
    
    # 2. 半精度支持（只在cuda可用）
    if half_model_dtype:
        model.half()
    print("Current dtype:", next(model.parameters()).dtype)
    print("Device:", next(model.parameters()).device)
    
    # 3. 设置随机种子（可选）
    # set random seed to guarantee reproducibility
    set_random_seed(seed)
    
    
    # 4. 处理 pad_token
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        print("原模型中没有pad_token,需要自己添加。")
        # 支持特殊的 WizardMath-70B
        # set the pad_token of pretrained and finetuned tokenizer
        # note that WizardMath-70B-V1.0 adds two tokens {"<pad>": 32000, "[PAD]": 32001} with (32002, 8192) token embedding size
        # therefore, for WizardMath-70B-V1.0, we add one distinct pad_token "<pad>[PAD]" to reshape the token embedding size to (32001, 8192)
        if "WizardMath-70B-V1.0" in model_name:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>[PAD]"),
                model=model,
                tokenizer=tokenizer,
            )
        else:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=model,
                tokenizer=tokenizer,
            )
    
    print("tokenizer vocab size:", tokenizer.vocab_size)
    print("model input embedding size:", model.get_input_embeddings().weight.shape)
    print("model lm_head size:", model.lm_head.weight.shape)
    
    print("模型加载完毕！")
    
    return model, tokenizer


def get_gsm8k_samples(n=20, split="test", field="question+answer", use_prompt=True):
    """
    加载 gsm8k 的前 n 条文本数据，返回文本列表。
    参数:
        n: 采样数量
        split: 'train' 或 'test'
        field: 'question' 或 'question+answer'
        use_prompt: 是否加格式前缀

    返回:
        samples: List[str]
    """

    # 下载 gsm8k
    ds = load_dataset("gsm8k", "main", split=split)
    samples = []
    for i in range(min(n, len(ds))):
        q = ds[i]['question'].strip()
        a = ds[i]['answer'].strip()
        if field == "question":
            text = q
        elif field == "question+answer":
            if use_prompt ==True:
                # 推荐格式，可灵活调整
                text = f"Question: {q} Answer: {a}"
                # 或 text = f"Q: {q}\nA: {a}"
            else:
                text = q + " " + a
        else:
            raise ValueError("field must be 'question' or 'question+answer'")
        samples.append(text)
    return samples
    
    


def get_humaneval_samples(n=20, split="test", field="prompt+solution", use_prompt=True):
    """
    加载 HumanEval 的前 n 条样本，返回文本列表。
    参数:
        n: 采样数量
        split: 通常只有'test'
        field: "prompt" 或 "prompt+solution"
        use_prompt: 是否加前缀
    返回:
        samples: List[str]
    """
    ds = load_dataset("openai_humaneval", split=split)
    samples = []
    for i in range(min(n, len(ds))):
        prompt = ds[i]["prompt"].strip()
        solution = ds[i].get("canonical_solution", "").strip()
        if field == "prompt":
            text = prompt
        elif field == "prompt+solution":
            if use_prompt:
                text = f"# Prompt\n{prompt}\n# Solution\n{solution}"
            else:
                text = prompt + "\n" + solution
        else:
            raise ValueError("field must be 'prompt' or 'prompt+solution'")
        samples.append(text)
    return samples



def collect_all_linear_activations(model, tokenizer, sample_texts, batch_size=1, device=None, verbose=True):
    """
    收集模型所有 Linear 层的输入张量 X，支持多文本输入（自动批量处理）。
    返回: activations_dict, module_name_map
    activations_dict: {module_name: [X1, X2, ...]}  # 每个层名对应输入序列列表
    module_name_map: {module_name: module_object}   # 便于后续查找权重
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    activations_dict = {}
    module_name_map = {}

    # 1. 注册所有 Linear 层 hook
    hooks = []
    def register_hooks(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # 如果是 Linear 层，则注册 hook
            if isinstance(child, nn.Linear):
                module_name_map[full_name] = child
                activations_dict[full_name] = []
                def save_input(name):
                    def hook(mod, inp, out):
                        # inp 是 tuple，取第一个元素
                        activations_dict[name].append(inp[0].detach().cpu())
                    return hook
                hooks.append(child.register_forward_hook(save_input(full_name)))
            else:
                register_hooks(child, full_name)
    register_hooks(model)

    # 2. 批量输入文本，触发 forward，收集激活
    # 建议分 batch 跑，避免显存爆炸
    total = len(sample_texts)
    for idx in range(0, total, batch_size):
        batch_texts = sample_texts[idx: idx+batch_size]
        # 转 tensor
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        if verbose:
            print(f"Processed {min(idx+batch_size, total)} / {total}")

    # 3. 移除所有 hooks，防止内存泄漏
    for h in hooks:
        h.remove()

    return activations_dict, module_name_map





def compute_wanda_metric(model, activations_dict, module_name_map, save_numpy=True):
    """
    计算每个 Linear 层的 Wanda 指标矩阵: |Wij| * ||Xj||2
        - X_j：指第 j 个 in_features 通道，合并所有 batch 和 seq 样本，再求 2 范数
        - 权重矩阵 W：shape [out_features, in_features]
    输入:
        model: pytorch model
        activations_dict: {module_name: [X1, X2, ...]}  # 每个X shape = [batch, seq_len, in_features]
        module_name_map: {module_name: module_object}

    输出:
        wanda_metric_dict: {module_name: wanda_matrix}  # 每个 wanda_matrix shape = [out_features, in_features]
    """
    
    wanda_metric_dict = {}

    for name, X_list in activations_dict.items():
        if len(X_list) == 0:
            print(f"[警告] 层 {name} 没有采集到任何输入！")
            continue
    
        # X_list 里每个 X shape: [batch, seq, in_features]（通常 batch=1，seq=任意）
        # # 1. 拼接所有采样：得到 [N, S, D]
        # X_cat = torch.cat(X_list, dim=0)  # [N, S, D]，N为所有采集batch数量
        # # 2. 展平成一个大样本：合并 batch 和 seq 维，得到 [total_tokens, in_features]
        # X_cat_flat = X_cat.reshape(-1, X_cat.shape[-1])  # [total_tokens, in_features]
        
        # 先 flatten 每个样本
        X_flattened_list = [x.reshape(-1, x.shape[-1]) for x in X_list]  # 适配不等长
        X_cat_flat = torch.cat(X_flattened_list, dim=0)  # [total_tokens, in_features]


        # 3. 对每个 in_features 列（通道）做 2 范数，shape [in_features]
        X_col_norm = torch.norm(X_cat_flat, p=2, dim=0)  # 每列合并后做 norm
        
        # 4. 取权重 |W|，shape [out_features, in_features]
        W = module_name_map[name].weight.data.detach().cpu().abs()  # [out_features, in_features]
        
        # 5. Wanda 指标（逐列乘法）：[out_features, in_features]
        wanda_matrix = W * X_col_norm.unsqueeze(0)  # 广播：每列都乘 norm
    
        if save_numpy:
            wanda_metric_dict[name] = wanda_matrix.numpy()
        else:
            wanda_metric_dict[name] = wanda_matrix
    
        print(f"{name}: W shape {W.shape}, X_col_norm shape {X_col_norm.shape}, Wanda shape {wanda_matrix.shape}")
        
    return wanda_metric_dict



def save_metric_results(save_path, metric_dict):
    """
    保存 wanda 指标结果到 npz 文件。
    参数:
        save_path: 文件名，如 'wizardmath13b_wanda_metrics.npz'
        metric_dict: {module_name: wanda_matrix}
    """
    # 确保输出目录存在
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 用 np.savez 保存所有模块
    np.savez(save_path, **metric_dict)
    print(f"保存 Wanda 指标到: {save_path}")




def analyze_and_save_wanda_metrics(
    model_name,
    get_samples_fn,        # 数据采样函数
    n_samples=20,
    sample_kwargs=None,    # 采样函数额外参数dict
    batch_size=1,
    save_folder="activations",
    dataset_name=None,
    # cache_dir=None,
    half_model_dtype=False,
    seed=0,
    verbose=True
):
    """
    一站式收集并保存 LLM 所有 Linear 层 Wanda 指标
    参数:
        model_name: 模型名或路径
        get_samples_fn: 采样函数，如 get_gsm8k_samples
        n_samples: 样本数
        sample_kwargs: 采样函数其它参数（如 split/field/use_prompt）
        batch_size: 前向 batch 大小
        save_path: 输出文件名，自动用 model_name 生成
        cache_dir: 模型缓存目录
        half_model_dtype: 是否用半精度
        seed: 随机种子
        verbose: 是否打印进度
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    # 1. 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name, half_model_dtype=half_model_dtype, seed=seed
    )

    # 2. 采样文本数据
    sample_texts = get_samples_fn(n=n_samples, **sample_kwargs)
    if verbose:
        print(f"采样数据条数: {len(sample_texts)}")

    # 3. 收集 activations
    activations_dict, module_map = collect_all_linear_activations(
        model, tokenizer, sample_texts, batch_size=batch_size, verbose=verbose
    )

    # 4. 计算 Wanda 指标
    wanda_metric_dict = compute_wanda_metric(model, activations_dict, module_map)

    # 5. 自动组织保存路径
    if dataset_name is None:
        dataset_name = sample_kwargs.get('dataset_name', 'unknown')
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    save_dir = os.path.join(save_folder, dataset_name, f"n{n_samples}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{safe_model_name}_wanda_metrics.npz")

    save_metric_results(save_path, wanda_metric_dict)


    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"全部流程结束！已保存至 {save_path}")
    return save_path




def main_wanda_activation_analysis():
    # WizardMath-13B-V1.0 + gsm8k
    analyze_and_save_wanda_metrics(
        model_name="WizardMath-13B-V1.0",
        get_samples_fn=get_gsm8k_samples,
        n_samples=64,
        sample_kwargs={'split': 'test', 'field': 'question+answer', 'use_prompt': True},
        batch_size=1,
        save_folder="activations",
        dataset_name="gsm8k"
    )

    # llama-2-13b-code-alpaca + humaneval
    analyze_and_save_wanda_metrics(
        model_name="llama-2-13b-code-alpaca",
        get_samples_fn=get_humaneval_samples,
        n_samples=64,
        sample_kwargs={'split': 'test', 'field': 'prompt+solution', 'use_prompt': True},
        batch_size=1,
        save_folder="activations",
        dataset_name="humaneval"
    )

if __name__ == "__main__":
    main_wanda_activation_analysis()
