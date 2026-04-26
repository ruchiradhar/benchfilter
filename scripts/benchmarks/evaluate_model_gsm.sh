#!/bin/bash

#The partition is the queue you want to run on. standard is gpu and can be ommitted. Check updates, mem-per-gpu
#SBATCH -p gpu --gres=gpu:a100:1    

#SBATCH --exclude=hendrixgpu04fl   

#SBATCH --mem=120G    

#SBATCH --job-name=evaluate_models_gsm_150-200                                                                                                                            

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
# declare -a model_names=("Tsunami-th/Tsunami-1.0-14B-Instruct" "Quazim0t0/Rune-14b" "sometimesanotion/Lamarck-14B-v0.7-rc4" "prithivMLmods/Porpoise-Opus-14B-Exp" "CombinHorizon/Josiefied-abliteratedV4-Qwen2.5-14B-Inst-BaseMerge-TIES" "prithivMLmods/Dinobot-Opus-14B-Exp" "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2" "sometimesanotion/Lamarck-14B-v0.7-Fusion" "Quazim0t0/SuperNova14b" "prithivMLmods/Messier-Opus-14B-Elite7" "Quazim0t0/Lineage-14B" "YOYO-AI/ZYH-LLM-Qwen2.5-14B-V3" "prithivMLmods/Pegasus-Opus-14B-Exp" "djuna/Q2.5-Veltha-14B-0.5" "braindao/Qwen2.5-14B-Instruct" "YOYO-AI/Qwen2.5-14B-YOYO-V4-p2" "sometimesanotion/LamarckInfusion-14B-v2-lo" "sometimesanotion/LamarckInfusion-14B-v3" "Qwen/Qwen2.5-14B-Instruct-1M" "notbdq/Qwen2.5-14B-Instruct-1M-GRPO-Reasoning" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.8" "Quazim0t0/MFGRIMM-14B" "sometimesanotion/Qwenvergence-14B-v11" "sometimesanotion/LamarckInfusion-14B-v2-hi" "YOYO-AI/Qwen2.5-14B-it-restore")
# Run 3 is second 25 of 74 under 14B in top 200
# declare -a model_names=("sometimesanotion/Qwenvergence-14B-v10" "CombinHorizon/huihui-ai-abliteratedV2-Qwen2.5-14B-Inst-BaseMerge-TIES" "RDson/WomboCombo-R1-Coder-14B-Preview" "Quazim0t0/MFDOOM-14B" "jpacifico/Chocolatine-2-14B-Instruct-v2.0b3" "Quazim0t0/Nova-14b-sce" "Quazim0t0/Geedorah-14B" "v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno" "Quazim0t0/Alice-14B" "Quazim0t0/NovaScotia-14b-stock" "Quazim0t0/Rosemary-14b" "Quazim0t0/ODB-14b-sce" "jpacifico/Chocolatine-2-14B-Instruct-v2.0.3" "nbeerbower/Qwen2.5-Gutenberg-Doppel-14B" "Qwen/Qwen2.5-14B-Instruct" "Quazim0t0/Ponder-14B-linear" "jpacifico/Chocolatine-2-14B-Instruct-v2.0b2" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v3" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3-model_stock" "Quazim0t0/caramel-14B" "Quazim0t0/Wendy-14B" "sometimesanotion/Qwenvergence-14B-v12-Prose-DS" "Quazim0t0/time-14b-stock" "sometimesanotion/Lamarck-14B-v0.6" "sometimesanotion/ChocoTrio-14B-v1")
# Run 4 is last 10 of 74 under 14B in top 200
# declare -a model_names=("Quazim0t0/1up-14b" "tensopolis/qwen2.5-14b-tensopolis-v1" "sometimesanotion/Lamarck-14B-v0.7-rc1" "v000000/Qwen2.5-Lumen-14B" "prithivMLmods/Sombrero-Opus-14B-Sm5" "sthenno-com/miscii-14b-0130" "sometimesanotion/Qwenvergence-14B-v13-Prose-DS" "Quazim0t0/Vine-14b-sce" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3" "Quazim0t0/Casa-14b-sce")
# Run 5 is first 25 of 67 under 20B in 201-301
# declare -a model_names=("Quazim0t0/Mithril-14B-sce" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v5" "Sakalti/ultiima-14B-v0.2" "prithivMLmods/Primal-Opus-14B-Optimus-v2" "v000000/Qwen2.5-14B-Gutenberg-1e-Delta" "hotmailuser/QwenSlerp2-14B" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v7" "Quazim0t0/Loke-14B-sce" "Quazim0t0/mosaic-14b-sce" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v9-stock" "pankajmathur/orca_mini_v9_2_14B" "sometimesanotion/Lamarck-14B-v0.6-model_stock" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v6" "CultriX/Qwen2.5-14B-ReasoningMerge" "Lunzima/NQLSG-Qwen2.5-14B-OriginalFusion" "sometimesanotion/Qwenvergence-14B-v0.6-004-model_stock" "Quazim0t0/Oasis-14B-ties" "prithivMLmods/Cygnus-II-14B" "Quazim0t0/GuiltySpark-14B-ties" "Quazim0t0/Casa-14b-sce" "Sakalti/ultiima-14B-v0.3" "ehristoforu/fp4-14b-v1-fix" "FINGU-AI/Chocolatine-Fusion-14B" "hotmailuser/QwenSlerp-14B" "Quazim0t0/Adamant-14B-sce")
# Run 6 is second 25 of 67 under 20B in 201-301
# declare -a model_names=("Quazim0t0/Phi4Basis-14B-sce" "Quazim0t0/bloom-14b-stock" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3-Prose01" "sometimesanotion/Qwentessential-14B-v1" "Quazim0t0/Halo-14B-sce" "prithivMLmods/Calcium-Opus-14B-Elite2" "SicariusSicariiStuff/Impish_QWEN_14B-1M" "Quazim0t0/Sake-20b" "sometimesanotion/Qwenvergence-14B-qv256" "prithivMLmods/Gaea-Opus-14B-Exp" "prithivMLmods/Eridanus-Opus-14B-r999" "YOYO-AI/Qwen2.5-14B-YOYO-1005" "YOYO-AI/Qwen2.5-14B-YOYO-latest" "prithivMLmods/Calcium-Opus-14B-Elite" "prithivMLmods/Magellanic-Opus-14B-Exp" "YOYO-AI/Qwen2.5-14B-YOYO-1010" "hotmailuser/QwenStock1-14B" "YOYO-AI/Qwen2.5-14B-YOYO-1005-v2" "YOYO-AI/Qwen2.5-14B-YOYO-0510-v2" "YOYO-AI/Qwen2.5-14B-YOYO-1010-v2" "qingy2024/Qwen2.5-Math-14B-Instruct-Preview" "CultriX/Qwen2.5-14B-HyperMarck-dl" "Quazim0t0/mocha-14B" "CultriX/Qwen2.5-14B-Brocav3" "CultriX/Qwen2.5-14B-Brocav6")
# Run 7 is last 17 of 67 under 20B in 201-301
# declare -a model_names=("YOYO-AI/ZYH-LLM-Qwen2.5-14B" "sometimesanotion/Qwenvergence-14B-v9" "hotmailuser/QwenSlerp3-14B" "CultriX/Qwen2.5-14B-Hyperionv3" "prithivMLmods/Tucana-Opus-14B-r999" "prithivMLmods/Calcium-Opus-14B-Elite-Stock" "CultriX/Qwen2.5-14B-Hyperionv5" "prithivMLmods/Volans-Opus-14B-Exp" "YOYO-AI/Qwen2.5-14B-YOYO-SCE" "YOYO-AI/Qwen2.5-14B-YOYO-0505" "YOYO-AI/Qwen2.5-14B-YOYO-0805" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.6" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v7-rebase" "CultriX/Qwen2.5-14B-Brocav7" "Sakalti/ultiima-14B" "sometimesanotion/Qwentinuum-14B-v6" "Quazim0t0/Dyson-14b")
# Run 8 is first 25 of 65 under 20B in 301-401
# declare -a model_names=("Quazim0t0/SZA-14B-sce" "qingy2024/Fusion4-14B-Instruct" "allknowingroger/QwenSlerp6-14B" "rombodawg/Rombos-LLM-V2.5-Qwen-14b" "allura-org/TQ2.5-14B-Aletheia-v1" "prithivMLmods/Galactic-Qwen-14B-Exp1" "sometimesanotion/Lamarck-14B-v0.6-002-model_stock" "prithivMLmods/Coma-II-14B" "Quazim0t0/tesseract-14b-stock" "prithivMLmods/Sombrero-Opus-14B-Sm4" "allknowingroger/QwenSlerp5-14B" "qingy2024/Qwen2.5-Math-14B-Instruct-Alpha" "sometimesanotion/Qwentinuum-14B-v5" "prithivMLmods/Evac-Opus-14B-Exp" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v4" "CultriX/Qwen2.5-14B-BrocaV9" "prithivMLmods/Sombrero-Opus-14B-Sm1" "unsloth/phi-4-unsloth-bnb-4bit" "Quazim0t0/Sumatra-20b" "sometimesanotion/Qwenvergence-14B-v8" "mrm8488/phi-4-14B-grpo-gsm8k-3e" "sometimesanotion/Qwentinuum-14B-v3" "sometimesanotion/Qwentinuum-14B-v7" "allura-org/TQ2.5-14B-Neon-v1" "CultriX/Qwen2.5-14B-partialmergept1" "prithivMLmods/Epimetheus-14B-Axo")
# Run 9 is next 25 models of 65 under 20B in 301-401
declare -a model_names=("mrm8488/phi-4-14B-grpo-limo" "unsloth/phi-4-bnb-4bit" "prithivMLmods/Sombrero-Opus-14B-Sm2" "sometimesanotion/Qwenvergence-14B-v6-Prose" "internlm/internlm2_5-20b-chat" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v9" "CultriX/Qwen2.5-14B-Ultimav2" "prithivMLmods/Calcium-Opus-14B-Elite3" "prithivMLmods/Gauss-Opus-14B-R999" "allknowingroger/QwenSlerp4-14B" "CultriX/Qwen2.5-14B-MegaMerge-pt2" "CultriX/Qwen2.5-14B-MergeStock" "CultriX/SeQwence-14Bv2" "sometimesanotion/Qwentinuum-14B-v6-Prose" "CultriX/SeQwence-14Bv3" "CultriX/Qwen2.5-14B-Emergedv3" "sometimesanotion/Qwentinuum-14B-v013" "CultriX/SeQwence-14Bv1" "prithivMLmods/Calcium-Opus-14B-Elite2-R1" "CultriX/Qwestion-14B" "netease-youdao/Confucius-o1-14B" "Quazim0t0/Venti-20b" "sometimesanotion/Qwentinuum-14B-v8" "CultriX/SeQwence-14B-EvolMergev1" "allknowingroger/QwenStock2-14B")


# Specify the task name  
task_name=mgsm_direct 

# Specify languages required
# required languages: English, Spanish, French, German, Swahili, Bengali, Chinese, Japanese.
# required language codes: en, es, fr, de, sw, bn, zh, ja

for model_name in "${model_names[@]}"
do
    lm_eval --model vllm \
    --model_args pretrained=$model_name,max_model_len=12000 \
        --tasks $task_name \
        --device cuda:0 \
        --batch_size 8 \
        --log_samples \
        --output_path ../../results/gsm/
done 

