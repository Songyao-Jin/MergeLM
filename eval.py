# from human_eval.evaluation import evaluate_functional_correctness

# result_file = "save_gen_codes_results/human_eval/WizardLM-13B-V1.2_inference_mask_0.0.jsonl"

# results = evaluate_functional_correctness(result_file, k=[1])
# print("Accuracy (pass@1):", results["pass@1"])




# import json
# import sys

# # 你原来的结果路径（可能是 save_gen_results_folder + ".jsonl"）
# input_path = "save_gen_codes_results/mbpp/WizardLM-13B-V1.2_inference_mask_0.0.jsonl"
# # 输出标准格式结果路径
# output_path = "save_gen_codes_results/mbpp/WizardLM-13B-V1.2_mbpp_standard.jsonl"

# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# with open(output_path, "w", encoding="utf-8") as fout:
#     for i, completions in enumerate(data):
#         if not completions:
#             continue
#         fout.write(json.dumps({
#             "task_id": 11 + i,  # MBPP 从 11 开始编号
#             "code": completions[0]  # 只取第一条
#         }) + "\n")

# print(f"Converted to standard format and saved to {output_path}")




import json

jsonl_path = "/data/songyao/MergeLM/save_gen_codes_results/mbpp/WizardLM-13B-V1.2_mbpp_standard.jsonl"
json_path = "/data/songyao/MergeLM/save_gen_codes_results/mbpp/WizardLM-13B-V1.2_mbpp_standard.json"

# 读取 JSONL
with open(jsonl_path, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

# 写入 JSON（标准 list 格式）
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)


# import json
# import traceback

# def run_code_with_assertions(candidate_code: str, setup_code: str, test_list: list[str]) -> bool:
#     """尝试执行代码和测试，捕获异常，如果全部通过就返回 True"""
#     global_namespace = {}
#     try:
#         exec(setup_code, global_namespace)
#         exec(candidate_code, global_namespace)
#         for test_case in test_list:
#             exec(test_case, global_namespace)
#         return True
#     except Exception:
#         traceback.print_exc()
#         return False

# # === 加载数据 ===
# # 你的生成结果文件（标准格式）
# generated_file = "save_gen_codes_results/mbpp/WizardLM-13B-V1.2_mbpp_standard.jsonl"
# # 对应 MBPP 的问题文件
# problem_file = "/data/songyao/MergeLM/math_code_data/mbpp.test.jsonl"  # 修改为你自己的路径！

# # 加载生成结果
# generated = {}
# with open(generated_file, "r", encoding="utf-8") as fin:
#     for line in fin:
#         record = json.loads(line)
#         generated[str(record["task_id"])] = record["completion"]

# # 加载 ground truth
# problems = {}
# with open(problem_file, "r", encoding="utf-8") as fin:
#     for line in fin:
#         problem = json.loads(line)
#         problems[str(problem["task_id"])] = problem

# # === 评估 ===
# passed = 0
# total = 0
# for task_id, problem in problems.items():
#     if task_id not in generated:
#         continue
#     candidate_code = generated[task_id]
#     setup_code = problem["test_setup_code"]
#     test_cases = problem["test_list"]
#     success = run_code_with_assertions(candidate_code, setup_code, test_cases)
#     passed += int(success)
#     total += 1

# accuracy = passed / total * 100
# print(f"MBPP pass@1 accuracy: {accuracy:.2f}% ({passed}/{total})")


