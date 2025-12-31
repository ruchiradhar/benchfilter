import torch
import datasets
from datasets import load_dataset, get_dataset_config_names  
import pdb
import json
import re 

def unpack_result(model_result, benchmark_name):
    results = {}
    if "__" in benchmark_name:
        benchmark_name = benchmark_name.split("__")[-1]
    latest_result = model_result["latest"]
    for lr in latest_result:
        if "acc_norm" in lr:
            result_str = "acc_norm"
        elif "acc" in lr:
            result_str = "acc"
        elif "exact_match" in lr:
            result_str = "exact_match"
        elif "prompt_level_strict_acc" in lr:
            result_str = "prompt_level_strict_acc"
        else:
            print (lr)
            raise ValueError("No known result string found in lr")

        # results.update({benchmark_name+"_"+str(lr["doc_id"]): lr[result_str]})
        results[benchmark_name+"_"+str(lr["doc_id"])] = lr[result_str]

    return results

def contains_size_greater_than(text, threshold=30):
    """
    Check if a string contains a model size (e.g., '14B', '72B') greater than threshold.
    
    Args:
        text: The string to search
        threshold: The size threshold to compare against (default: 30)
    
    Returns:
        bool: True if any size greater than threshold is found, False otherwise
    """
    # Pattern matches number followed by 'B' or 'b' (case-insensitive)
    pattern = r'(\d+(?:\.\d+)?)[Bb]'
    
    matches = re.findall(pattern, text)

    if len(matches) == 0:
        return True
    
    for match in matches:
        size = float(match)
        if size > threshold:
            return True
    
    return False

dataset = load_dataset("open-llm-leaderboard/contents", split="train")

dataset_df = dataset.to_pandas()
ranked_df = dataset_df.sort_values(by='Average ⬆️', ascending=False, ignore_index=True)
ranked_df_full_name = ranked_df["fullname"].tolist()
top200_ranked_df_full_names = ranked_df_full_name[:300]   

pattern = r'(\d+(?:\.\d+)?)[Bb]'
filtered_model_names = []

for model_name in top200_ranked_df_full_names:
    if contains_size_greater_than(model_name, threshold=20):
        continue
    filtered_model_names.append(model_name)

print (filtered_model_names) 
print ("there are ", len(filtered_model_names), "models under 20B in the top 200")
print ('" "'.join(filtered_model_names))
print ("~~~~~~~~~~~~~~~~~~~~")
print ('" "'.join(filtered_model_names[50:100]))       



# all_model_results = []
# idx = 0 
# for model_name in top200_ranked_df_full_names:
#     model_name = model_name.replace("/", "__")+"-details"
#     if model_name == "JungZoona__T3Q-Qwen2.5-14B-Instruct-1M-e3-details":
#         continue
#     repo_id = f"open-llm-leaderboard/{model_name}"
#     model_config_names = get_dataset_config_names(repo_id)
#     results_across_benchmarks = {}
#     for model_config_name in model_config_names:
#         try:
#             model_result = load_dataset(repo_id, model_config_name)
#         except:
#             print(f"Skipping {model_name} {model_config_name} due to load error")
#             continue
#         unpack_model_result = unpack_result(model_result, model_config_name)
#         results_across_benchmarks.update(unpack_model_result)

#     all_model_results.append({"subject_id": model_name, "responses": results_across_benchmarks}) 

#     if idx%10 == 0:
#         print(f"Processed {idx} models")

#     idx += 1


# with open('/home/tfv783/benchfilter/data/raw/open_llm_benchmark_top200_results.jsonl', 'w', encoding='utf-8') as f:
#     for entry in all_model_results:
#         json_line = json.dumps(entry)
#         f.write(json_line + '\n')



"""
lm_eval --model vllm \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks mgsm_direct \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path ./results/

lm_eval --model vllm --model_args pretrained=EleutherAI/gpt-j-6B --tasks mgsm_direct --device cuda:0 --batch_size 8 --log_samples --output_path ./results/
"""