#!/bin/bash

# for ds in gsm8k MATH human_eval mbpp
# do
#   echo "Running inference on $ds ..."
#   python inference_llms_instruct_math_code.py \
#     --dataset_name $ds \
#     --finetuned_model_name llama2-13b-math-code-alignment-sparseft_dataSize512each \
#     --tensor_parallel_size 4 \
#     --weight_mask_rate 0.0
# done


for ds in gsm8k MATH
do
  echo "Running inference on $ds ..."
  python inference_llms_instruct_math_code.py \
    --dataset_name $ds \
    --finetuned_model_name llama2-13b-math-code-alignment-sparseft_dataSize1024each \
    --tensor_parallel_size 4 \
    --weight_mask_rate 0.0
done