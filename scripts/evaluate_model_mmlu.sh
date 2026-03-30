#!/bin/bash

#The partition is the queue you want to run on. standard is gpu and can be ommitted. Check updates, mem-per-gpu
#SBATCH -p gpu --gres=gpu:a100:1    

#SBATCH --exclude=hendrixgpu04fl   

#SBATCH --mem=120G    

#SBATCH --job-name=evaluate_models_mmlu_150-200                                                                                                                            

#SBATCH -o /home/nwq388/projects/benchfilter/logs/log-%j.%x.out                       

#SBATCH --ntasks=1

#We expect that our program should not run longer than 2 days 
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00

# mail alert at start, end and abortion of execution   
##SBATCH --mail-type=ALL

# send mail to this address

#Skipping many options! see man sbatch 
# From here on, we can start our program

# setting HF token
export HF_TOKEN=***REMOVED-HF-TOKEN***

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.8  
eval "$(conda shell.bash hook)"
conda activate benchfilterenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn    
 
#your script, in this case: write the hostname and the ids of the chosen gpus and the status of the GPU.  
hostname  

# Specify model names
# Run 1 was 21 models under 14B in top 100
# Run 2 is top 25 of 74 under 14B in top 200
declare -a model_names=("Tsunami-th/Tsunami-1.0-14B-Instruct" "Quazim0t0/Rune-14b" "sometimesanotion/Lamarck-14B-v0.7-rc4" "prithivMLmods/Porpoise-Opus-14B-Exp" "CombinHorizon/Josiefied-abliteratedV4-Qwen2.5-14B-Inst-BaseMerge-TIES" "prithivMLmods/Dinobot-Opus-14B-Exp" "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2" "sometimesanotion/Lamarck-14B-v0.7-Fusion" "Quazim0t0/SuperNova14b" "prithivMLmods/Messier-Opus-14B-Elite7" "Quazim0t0/Lineage-14B" "YOYO-AI/ZYH-LLM-Qwen2.5-14B-V3" "prithivMLmods/Pegasus-Opus-14B-Exp" "djuna/Q2.5-Veltha-14B-0.5" "braindao/Qwen2.5-14B-Instruct" "YOYO-AI/Qwen2.5-14B-YOYO-V4-p2" "sometimesanotion/LamarckInfusion-14B-v2-lo" "sometimesanotion/LamarckInfusion-14B-v3" "Qwen/Qwen2.5-14B-Instruct-1M" "notbdq/Qwen2.5-14B-Instruct-1M-GRPO-Reasoning" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.8" "Quazim0t0/MFGRIMM-14B" "sometimesanotion/Qwenvergence-14B-v11" "sometimesanotion/LamarckInfusion-14B-v2-hi" "YOYO-AI/Qwen2.5-14B-it-restore")
 
# Specify task_name
# global_mmlu_<language_code>

# Specify languages required
#required languages: English, Spanish, French, German, Swahili, Bengali, Chinese, Japanese.
declare -a languages=("en" "es" "fr" "de" "sw" "bn" "zh" "ja")

for model_name in "${model_names[@]}"
do
    for lang in "${languages[@]}"
    do
        task_name="global_mmlu_${lang}"
        lm_eval --model vllm \
            --model_args pretrained=$model_name,max_model_len=12000 \
            --tasks $task_name \
            --device cuda:0 \
            --batch_size 8 \
            --log_samples \
            --output_path ../results/mmlu/
    done
done
