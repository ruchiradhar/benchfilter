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
# declare -a model_names=("braindao/Qwen2.5-14B-Instruct" "YOYO-AI/Qwen2.5-14B-YOYO-V4-p2" "sometimesanotion/LamarckInfusion-14B-v2-lo" "sometimesanotion/LamarckInfusion-14B-v3" "notbdq/Qwen2.5-14B-Instruct-1M-GRPO-Reasoning" "Qwen/Qwen2.5-14B-Instruct-1M" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.8" "Quazim0t0/MFGRIMM-14B" "sometimesanotion/Qwenvergence-14B-v11" "sometimesanotion/LamarckInfusion-14B-v2-hi" "YOYO-AI/Qwen2.5-14B-it-restore" "sometimesanotion/Qwenvergence-14B-v10" "CombinHorizon/huihui-ai-abliteratedV2-Qwen2.5-14B-Inst-BaseMerge-TIES" "RDson/WomboCombo-R1-Coder-14B-Preview" "Quazim0t0/MFDOOM-14B" "jpacifico/Chocolatine-2-14B-Instruct-v2.0b3" "Quazim0t0/Nova-14b-sce" "Quazim0t0/Geedorah-14B" "v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno" "Quazim0t0/Alice-14B" "Quazim0t0/NovaScotia-14b-stock" "Quazim0t0/Rosemary-14b" "Quazim0t0/ODB-14b-sce" "jpacifico/Chocolatine-2-14B-Instruct-v2.0.3" "nbeerbower/Qwen2.5-Gutenberg-Doppel-14B" "Qwen/Qwen2.5-14B-Instruct" "Quazim0t0/Ponder-14B-linear" "jpacifico/Chocolatine-2-14B-Instruct-v2.0b2" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v3" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3-model_stock" "Quazim0t0/caramel-14B" "Quazim0t0/Wendy-14B" "sometimesanotion/Qwenvergence-14B-v12-Prose-DS" "Quazim0t0/time-14b-stock" "sometimesanotion/Lamarck-14B-v0.6" "sometimesanotion/ChocoTrio-14B-v1" "Quazim0t0/1up-14b" "tensopolis/qwen2.5-14b-tensopolis-v1" "sometimesanotion/Lamarck-14B-v0.7-rc1" "v000000/Qwen2.5-Lumen-14B" "prithivMLmods/Sombrero-Opus-14B-Sm5" "sthenno-com/miscii-14b-0130" "sometimesanotion/Qwenvergence-14B-v13-Prose-DS" "Quazim0t0/Vine-14b-sce" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3" "Quazim0t0/Casa-14b-sce" "Quazim0t0/Mithril-14B-sce" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v5" "Sakalti/ultiima-14B-v0.2" "prithivMLmods/Primal-Opus-14B-Optimus-v2")
# declare -a model_names=("JungZoona/T3Q-qwen2.5-14b-v1.0-e3" "JungZoona/T3Q-Qwen2.5-14B-Instruct-1M-e3" "wanlige/li-14b-v0.4" "prithivMLmods/Galactic-Qwen-14B-Exp2" "suayptalha/Lamarckvergence-14B" "suayptalha/Lix-14B-v0.1" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v9.1" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v9.2" "YOYO-AI/ZYH-LLM-Qwen2.5-14B-V4" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.7" "tensopolis/lamarckvergence-14b-tensopolis-v1" "wanlige/li-14b-v0.4-slerp0.1" "sthenno-com/miscii-14b-0218" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8" "tanliboy/lambda-qwen2.5-14b-dpo-test" "YOYO-AI/Qwen2.5-14B-1M-YOYO-V3" "Goekdeniz-Guelmez/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.9" "djuna/Q2.5-Veltha-14B" "YOYO-AI/Qwen2.5-14B-YOYO-V4-p1" "jpacifico/Chocolatine-14B-Instruct-DPO-v1.3" "sthenno-com/miscii-14b-1028")
# declare -a model_names=("sthenno-com/miscii-14b-1225" "prithivMLmods/Sombrero-Opus-14B-Elite5" "YOYO-AI/Qwen2.5-14B-YOYO-V4" "prithivMLmods/Equuleus-Opus-14B-Exp" "rombodawg/Rombos-LLM-V2.6-Qwen-14b" "nbeerbower/EVA-abliterated-TIES-Qwen2.5-14B" "sometimesanotion/LamarckInfusion-14B-v2" "sometimesanotion/LamarckInfusion-14B-v1" "Quazim0t0/Fugazi14b" "1024m/QWEN-14B-B100" "Sakalti/Saka-14B" "sthenno/tempestissimo-14b-0309" "prithivMLmods/Sombrero-Opus-14B-Elite6" "YOYO-AI/Qwen2.5-14B-YOYO-latest-V2" "Tsunami-th/Tsunami-1.0-14B-Instruct" "Quazim0t0/Rune-14b" "sometimesanotion/Lamarck-14B-v0.7-rc4" "prithivMLmods/Porpoise-Opus-14B-Exp" "prithivMLmods/Dinobot-Opus-14B-Exp" "CombinHorizon/Josiefied-abliteratedV4-Qwen2.5-14B-Inst-BaseMerge-TIES")
# declare -a model_names=("huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2" "sometimesanotion/Lamarck-14B-v0.7-Fusion" "Quazim0t0/SuperNova14b" "prithivMLmods/Messier-Opus-14B-Elite7" "Quazim0t0/Lineage-14B" "YOYO-AI/ZYH-LLM-Qwen2.5-14B-V3" "prithivMLmods/Pegasus-Opus-14B-Exp" "djuna/Q2.5-Veltha-14B-0.5" "braindao/Qwen2.5-14B-Instruct" "YOYO-AI/Qwen2.5-14B-YOYO-V4-p2" "sometimesanotion/LamarckInfusion-14B-v2-lo" "sometimesanotion/LamarckInfusion-14B-v3" "Qwen/Qwen2.5-14B-Instruct-1M" "notbdq/Qwen2.5-14B-Instruct-1M-GRPO-Reasoning" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.8" "Quazim0t0/MFGRIMM-14B" "sometimesanotion/Qwenvergence-14B-v11" "sometimesanotion/LamarckInfusion-14B-v2-hi" "YOYO-AI/Qwen2.5-14B-it-restore" "sometimesanotion/Qwenvergence-14B-v10" "CombinHorizon/huihui-ai-abliteratedV2-Qwen2.5-14B-Inst-BaseMerge-TIES" "RDson/WomboCombo-R1-Coder-14B-Preview" "Quazim0t0/MFDOOM-14B" "jpacifico/Chocolatine-2-14B-Instruct-v2.0b3" "Quazim0t0/Nova-14b-sce" "Quazim0t0/Geedorah-14B" "v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno" "Quazim0t0/Alice-14B" "Quazim0t0/NovaScotia-14b-stock" "Quazim0t0/Rosemary-14b" "Quazim0t0/ODB-14b-sce" "jpacifico/Chocolatine-2-14B-Instruct-v2.0.3" "nbeerbower/Qwen2.5-Gutenberg-Doppel-14B" "Qwen/Qwen2.5-14B-Instruct" "Quazim0t0/Ponder-14B-linear" "jpacifico/Chocolatine-2-14B-Instruct-v2.0b2" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v3" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3-model_stock" "Quazim0t0/caramel-14B" "Quazim0t0/Wendy-14B" "sometimesanotion/Qwenvergence-14B-v12-Prose-DS" "Quazim0t0/time-14b-stock" "sometimesanotion/Lamarck-14B-v0.6" "sometimesanotion/ChocoTrio-14B-v1" "Quazim0t0/1up-14b" "tensopolis/qwen2.5-14b-tensopolis-v1" "sometimesanotion/Lamarck-14B-v0.7-rc1" "v000000/Qwen2.5-Lumen-14B" "prithivMLmods/Sombrero-Opus-14B-Sm5" "sthenno-com/miscii-14b-0130" "sometimesanotion/Qwenvergence-14B-v13-Prose-DS" "Quazim0t0/Vine-14b-sce" "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3" "Quazim0t0/Casa-14b-sce")
# declare -a model_names=("Quazim0t0/SZA-14B-sce" "qingy2024/Fusion4-14B-Instruct" "allknowingroger/QwenSlerp6-14B" "rombodawg/Rombos-LLM-V2.5-Qwen-14b" "allura-org/TQ2.5-14B-Aletheia-v1" "prithivMLmods/Galactic-Qwen-14B-Exp1" "sometimesanotion/Lamarck-14B-v0.6-002-model_stock" "prithivMLmods/Coma-II-14B" "Quazim0t0/tesseract-14b-stock" "prithivMLmods/Sombrero-Opus-14B-Sm4" "allknowingroger/QwenSlerp5-14B" "qingy2024/Qwen2.5-Math-14B-Instruct-Alpha" "sometimesanotion/Qwentinuum-14B-v5" "prithivMLmods/Evac-Opus-14B-Exp" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v4" "CultriX/Qwen2.5-14B-BrocaV9" "prithivMLmods/Sombrero-Opus-14B-Sm1" "unsloth/phi-4-unsloth-bnb-4bit" "Quazim0t0/Sumatra-20b" "sometimesanotion/Qwenvergence-14B-v8")
# declare -a model_names=("mrm8488/phi-4-14B-grpo-gsm8k-3e" "sometimesanotion/Qwentinuum-14B-v3" "sometimesanotion/Qwentinuum-14B-v7" "allura-org/TQ2.5-14B-Neon-v1" "CultriX/Qwen2.5-14B-partialmergept1" "prithivMLmods/Epimetheus-14B-Axo" "mrm8488/phi-4-14B-grpo-limo" "unsloth/phi-4-bnb-4bit" "prithivMLmods/Sombrero-Opus-14B-Sm2" "sometimesanotion/Qwenvergence-14B-v6-Prose" "internlm/internlm2_5-20b-chat" "Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v9" "CultriX/Qwen2.5-14B-Ultimav2" "prithivMLmods/Calcium-Opus-14B-Elite3" "prithivMLmods/Gauss-Opus-14B-R999" "allknowingroger/QwenSlerp4-14B" "CultriX/Qwen2.5-14B-MegaMerge-pt2" "CultriX/Qwen2.5-14B-MergeStock" "CultriX/SeQwence-14Bv2" "sometimesanotion/Qwentinuum-14B-v6-Prose" "CultriX/SeQwence-14Bv3")
declare -a model_names=(
"CultriX/Qwen2.5-14B-Emergedv3"
"sometimesanotion/Qwentinuum-14B-v013"
"CultriX/SeQwence-14Bv1"
"prithivMLmods/Calcium-Opus-14B-Elite2-R1"
"CultriX/Qwestion-14B"
"netease-youdao/Confucius-o1-14B"
"Quazim0t0/Venti-20b"
"sometimesanotion/Qwentinuum-14B-v8"
"CultriX/SeQwence-14B-EvolMergev1"
"allknowingroger/QwenStock2-14B"
"Triangle104/DS-R1-Distill-Q2.5-14B-Harmony_V0.1"
"CultriX/Qwen2.5-14B-Wernickev3"
"Triangle104/Herodotos-14B"
"allknowingroger/QwenStock3-14B"
"CultriX/Qwen2.5-14B-Unity"
"prithivMLmods/Calcium-Opus-14B-Elite"
"Lunzima/NQLSG-Qwen2.5-14B-MegaFusion-v8.5"
"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
"Quazim0t0/GZA-14B-sce"
"Sakalti/SJT-14B"
)
 
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
            --model_args pretrained=$model_name \
            --tasks $task_name \
            --device cuda:0 \
            --batch_size 8 \
            --log_samples \
            --output_path ../results/mmlu/
    done
done
