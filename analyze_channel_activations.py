import os
import logging
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.load_config import cache_dir
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp
import seaborn as sns
import argparse


def get_activation_hook(activations, layer_idx, module_name):
    def hook(module, input, output):
        # Save output activations of the specific module in the given layer
        activations.append(output.detach().cpu())
        print("Added one new token.")
    return hook


def visualize_activations(activations, tokens, save_path, layer_number, module_name):
    os.makedirs(save_path, exist_ok=True)
    act_tensor = activations[0].squeeze(0)  # shape: (seq_len, hidden_size)

    for idx, token in enumerate(tokens):
        plt.figure(figsize=(12, 4))
        plt.bar(range(act_tensor[idx].shape[0]), act_tensor[idx].numpy())
        plt.title(f"Layer {layer_number} - {module_name} - Token: {token}")
        plt.xlabel("Channel")
        plt.ylabel("Activation")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"token_{idx}_{token}.png"))
        plt.close()

    # plot average activation across tokens
    avg_activation = act_tensor.mean(dim=0)
    plt.figure(figsize=(12, 4))
    plt.bar(range(avg_activation.shape[0]), avg_activation.numpy())
    plt.title(f"Layer {layer_number} - {module_name} - Average Activation")
    plt.xlabel("Channel")
    plt.ylabel("Average Activation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"average_activation.png"))
    plt.close()



def visualize_activations(activations, tokens, save_path, layer_number, module_name):
    os.makedirs(save_path, exist_ok=True)

    seq_len = None
    hidden_size = None
    if module_name in ["gate_proj", "up_proj", "down_proj"]:
        # 合并所有激活结果：每个 activation 是 (1, 1, hidden_size)
        # 输出 shape 应该是 (seq_len, hidden_size)
        act_tensor = torch.cat([act.view(-1, act.shape[-1]) for act in activations], dim=0)  # shape: (seq_len, hidden_size)
        seq_len, hidden_size = act_tensor.shape
    

    # for idx, token in enumerate(tokens):
    #     token_act = act_tensor[idx].numpy().reshape(1, -1)  # shape: (1, hidden_size)
    #     plt.figure(figsize=(12, 1.5))
    #     sns.heatmap(token_act, cmap="viridis", cbar=True, xticklabels=50, yticklabels=False)
    #     plt.title(f"Layer {layer_number} - {module_name} - Token: {token}", fontsize=10)
    #     plt.xlabel("Channel")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_path, f"token_{idx}_{token}.png"))
    #     plt.close()

    # Plot heatmap for all token activations (seq_len x hidden_size)
    plt.figure(figsize=(12, seq_len * 0.3 + 2))
    sns.heatmap(act_tensor.numpy(), cmap="viridis", cbar=True, xticklabels=50, yticklabels=tokens)
    plt.title(f"Layer {layer_number} - {module_name} - All Token Activations", fontsize=12)
    plt.xlabel("Channel")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"all_tokens_heatmap.png"))
    plt.close()

    # plot average activation across tokens
    avg_activation = act_tensor.mean(dim=0)
    plt.figure(figsize=(12, 4))
    plt.bar(range(avg_activation.shape[0]), avg_activation.numpy())
    plt.title(f"Layer {layer_number} - {module_name} - Average Activation")
    plt.xlabel("Channel")
    plt.ylabel("Average Activation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"average_activation.png"))
    plt.close()




def run_inference_for_one_sample(       args, prompt_dict, save_activation_dir):
    
    # if args.model_name == "WizardLM-13B-V1.2":
    #     args.model_name = "WizardLMTeam/WizardLM-13B-V1.2"
    # elif args.model_name == "WizardMath-13B-V1.0":
    #     args.model_name == "vanillaOVO/WizardMath-13B-V1.0"
    # elif args.model_name == "llama-2-13b-code-alpaca":
    #     args.model_name == "layoric/llama-2-13b-code-alpaca"
    
    try:
        # model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_name, cache_dir=cache_dir, device_map="auto")
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.model_name), device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.model_name))      
    except:
        print("wrong")
    
    
    print("Model loaded.")
    
    # # set the pad_token of pretrained and finetuned tokenizer
    # # note that WizardMath-70B-V1.0 adds two tokens {"<pad>": 32000, "[PAD]": 32001} with (32002, 8192) token embedding size
    # # therefore, for WizardMath-70B-V1.0, we add one distinct pad_token "<pad>[PAD]" to reshape the token embedding size to (32001, 8192)
    # if "WizardMath-70B-V1.0" in args.model_name:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token="<pad>[PAD]"),
    #         model=model,
    #         tokenizer=tokenizer,
    #     )
    # else:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token="[PAD]"),
    #         model=model,
    #         tokenizer=tokenizer,
    #     )
    
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    model.eval()
    # 改为半精度
    if args.half_model_dtype:
        model.half()
    print("Current dtype:", model.dtype)  # 一般是 torch.float32
    


    # Register hook
    activations = []
    target_layer = model.model.layers[args.layer_number]
    if hasattr(target_layer, "mlp") and hasattr(target_layer.mlp, args.proj_name):
        module = getattr(target_layer.mlp, args.proj_name)
    elif hasattr(target_layer, args.proj_name):
        module = getattr(target_layer, args.proj_name)
    else:
        raise ValueError(f"Module {args.proj_name} not found in layer {args.layer_number}")
    
    handle = module.register_forward_hook(get_activation_hook(activations, args.layer_number, args.proj_name))


    if args.prompt_type == "gsm8k":
        problem_prompt = get_math_task_prompt()
        inputs = problem_prompt.format(instruction=prompt_dict[args.prompt_type])
        
    elif args.prompt_type == "mbpp":
        prompt = f"\n{prompt_dict[args.prompt_type]['text']}\nTest examples:"
        for test_example in prompt_dict[args.prompt_type]['test_list']:
            prompt += f"\n{test_example}"
        prompt = prompt.replace('    ', '\t')
        inputs = generate_code_task_prompt(prompt)   
    
    print("Starting generation...")    
    input_ids = tokenizer(inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**input_ids, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None)
    print("Generation complete.")

    generated_ids = output_ids[0][input_ids["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(output_text)
    
    handle.remove()
    tokens = tokenizer.convert_ids_to_tokens(generated_ids, skip_special_tokens=True)
    visualize_activations(activations, tokens, save_activation_dir, args.layer_number, args.proj_name)
    
    # Save activations tensor
    activation_save_path = os.path.join(save_activation_dir, f"activations_layer{args.layer_number}_{args.proj_name}.pt")
    torch.save(activations, activation_save_path)
    print(f"Saved activations to {activation_save_path}")
    # Also save tokens and output text
    with open(os.path.join(save_activation_dir, "generated_text.txt"), "w", encoding="utf-8") as f:
        f.write(output_text)
    with open(os.path.join(save_activation_dir, "generated_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
    print("Saved generated text and tokens.")
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Analyze channel activations for Wizard models")
    parser.add_argument("--model_name", type=str, default="WizardLM-13B-V1.2", help="name of the language model",
                        choices=["WizardLM-7B-V1.0", "WizardLM-13B-V1.2", "WizardLM-70B-V1.0",
                                 "WizardMath-7B-V1.0", "WizardMath-13B-V1.0", "WizardMath-70B-V1.0",
                                 "WizardCoder-Python-7B-V1.0", "WizardCoder-Python-13B-V1.0", "WizardCoder-Python-34B-V1.0",
                                 "llama-2-13b-code-alpaca"])
    parser.add_argument("--half_model_dtype", action="store_true", default=False, help="whether to merge instruct model")
    parser.add_argument("--prompt_type", type=str, default="gsm8k", help="type of prompt to be used", choices=["gsm8k", "mbpp", "instruct"]) 
    parser.add_argument("--layer_number", type=int, default=1)
    parser.add_argument("--proj_name", type=str, required=True,
                        choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    prompt_dict = {
        "gsm8k": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",   #gsm8k中第一个问题
        "mbpp": {"task_id": 11, "text": "Write a python function to remove first and last occurrence of a given character from the string.", "code": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ", "test_list": ["assert remove_Occ(\"hello\",\"l\") == \"heo\"", "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"", "assert remove_Occ(\"PHP\",\"P\") == \"H\""], "test_setup_code": "", "challenge_test_list": ["assert remove_Occ(\"hellolloll\",\"l\") == \"helollol\"", "assert remove_Occ(\"\",\"l\") == \"\""]},   #mbpp里面的第一个问题
        # "instruct": "Please summarize the following paragraph:"
    }

    save_activation_dir = os.path.join("activation_analysis", args.prompt_type, args.model_name, f"layer_{args.layer_number}", args.proj_name)
    if args.half_model_dtype:
        save_activation_dir = os.path.join(save_activation_dir,"half_dtype")
    else:
        save_activation_dir = os.path.join(save_activation_dir,"normal_dtype")
    os.makedirs(save_activation_dir, exist_ok=True)
    
    run_inference_for_one_sample(args, prompt_dict, save_activation_dir)
    
        
        
        
        
        
        
        
        
    