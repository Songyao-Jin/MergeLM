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


# model = AutoModelForCausalLM.from_pretrained("merged_models/llama-2-13b-math-code-wanda-merge_hf", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("merged_models/llama-2-13b-math-code-wanda-merge_hf", use_fast=False)
# # model = AutoModelForCausalLM.from_pretrained("/data/songyao/models/llama-2-13b-code-alpaca", device_map="auto")
# # tokenizer = AutoTokenizer.from_pretrained("/data/songyao/models/llama-2-13b-code-alpaca", use_fast=False)


# # model, tokenizer = load_model_and_tokenizer("Llama-2-13b-hf", half_model_dtype=False, seed=0)


# print("fake tokenizer vocab size:", tokenizer.vocab_size)
# print("true tokenizer vocab size:", len(tokenizer))
# print("model input embedding size:", model.get_input_embeddings().weight.shape)
# print("model lm_head size:", model.lm_head.weight.shape)


# # Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens?
# prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nA robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\n\n### Response: Let's think step by step."
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
# # print(input_ids)

# output = model.generate(input_ids, max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.eos_token_id)
# decoded = tokenizer.decode(output[0], skip_special_tokens=True)
# print(decoded)




from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "merged_models/llama-2-13b-math-code-wanda-merge_hf"
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

# tokenizer.vocab_size = len(tokenizer)
# print("Fixed vocab size:", tokenizer.vocab_size)


print(tokenizer.get_vocab().get("[PAD]", None)) # 应该返回 32000
print(tokenizer.convert_tokens_to_ids("[PAD]")) # 也应返回 32000

# fixed_dir = "merged_models/llama-2-13b-math-code-wanda-merge_fixed"
# model.save_pretrained(fixed_dir)
# tokenizer.save_pretrained(fixed_dir)