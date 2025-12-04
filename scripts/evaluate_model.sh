#!/bin/bash

#The partition is the queue you want to run on. standard is gpu and can be ommitted. Check updates, mem-per-gpu
#SBATCH -p gpu --gres=gpu:a100:1    

#SBATCH --exclude=hendrixgpu04fl   

#SBATCH --mem=120G    

#SBATCH --job-name=evaluate_models_on_math_tasks                                                                                                                     

#SBATCH -o logs/log-%j.%x.out                     

#SBATCH --ntasks=1

#We expect that our program should not run longer than 2 days 
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00

# mail alert at start, end and abortion of execution
##SBATCH --mail-type=ALL

# send mail to this address

#Skipping many options! see man sbatch
# From here on, we can start our program

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate reasoning-consistency

#your script, in this case: write the hostname and the ids of the chosen gpus and the status of the GPU.
hostname
echo $CUDA_VISIBLE_DEVICES

model_name=EleutherAI/gpt-j-6B
task_name=mgsm_direct

lm_eval --model vllm \
    --model_args pretrained=$model_name \
    --tasks $task_name \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path ./results/