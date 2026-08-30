# BenchFilter

A research project applying Item Response Theory (IRT) to identify the most relevant subset of items/questions for evaluating Large Language Models, and to test how well IRT-based benchmark filtering generalizes across languages on translated multilingual benchmarks (MGSM, Global MMLU).

This repository contains the code needed to reproduce the full pipeline: running LLM evaluations, converting results into IRT-compatible response matrices, fitting 1PL/2PL/3PL IRT models, and running the downstream analyses. Raw evaluation outputs, fitted model artifacts, figures, and result tables are not included here — running the pipeline below regenerates them locally.

## Project Structure

```
benchfilter/
├── README.md                    # This file
├── LICENSE                      # License information
├── requirements.yaml            # Conda environment specification
├── .gitignore                   # Git ignore patterns
│
├── configs/                     # Configuration files
│   └── config._train.yaml
│
├── notebooks/                   # Jupyter notebooks
│   └── BasicsOfIRT.ipynb        # Introduction to IRT models (1PL/2PL/3PL/4PL)
│
├── src/                         # Shared source code
│   ├── irt_methods/
│   ├── llm_interface/
│   ├── evaluation/
│   └── utils/
│
└── scripts/
    ├── benchmarks/               # SLURM job scripts to run LLM evaluations (lm-evaluation-harness + vLLM)
    │   ├── evaluate_model_gsm.sh
    │   └── evaluate_model_mmlu.sh
    ├── irt/                      # Convert eval results to IRT format, fit 1PL/2PL/3PL/4PL models
    │   ├── convert_to_irt.py
    │   ├── run_convert_to_irt.sh
    │   ├── irt_onepl.py
    │   ├── irt_twopl.py
    │   ├── irt_threepl.py
    │   ├── irt_fourpl.py
    │   ├── irt_map.py            # MAP-based 2PL/3PL fitting (used for the paper's reported results)
    │   └── compute_elbo.py
    └── core_analysis/            # Analysis scripts producing the paper's reported statistics
        ├── gsm/, mmlu/           # per-benchmark rank-consistency / science-domain scripts
        ├── cross_model_concordance.py, cross_model_iif_concordance.py
        ├── dif_anchor.py, plot_dif_summary.py
        ├── param_decomposition.py, crosslingual_param_decomposition.py, plot_param_decomposition.py
        ├── subset_ranking_fidelity.py, subset_ranking_fidelity_map.py
        ├── split_half_reliability.py, split_half_vs_full.py
        ├── svi_vs_map_1pl.py, svi_vs_map_2pl_3pl.py, svi_vs_map_all.py
        ├── elbo_vs_map_loglik.py, leave_family_out_kendalls_w.py
        └── mean_accuracy_baseline.py, iif_theta_sensitivity.py, compression_sensitivity.py
```

`scripts/add_analysis/` mirrors `scripts/core_analysis/` and was an earlier exploratory pass (it also includes a 4PL variant that was not used in the final results); `scripts/core_analysis/` is what the paper's reported numbers are generated from.

## Overview

This project:
1. Runs LLM evaluations on two translated multilingual benchmarks — Multilingual GSM (MGSM) and Global MMLU — via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
2. Converts the raw evaluation outputs into per-language(/domain) item-response matrices.
3. Fits 1PL, 2PL, and 3PL IRT models to each benchmark slice.
4. Analyzes cross-lingual consistency of IRT parameters and the validity of IRT-filtered benchmark subsets across languages.

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f requirements.yaml
   conda activate benchfilterenv
   ```

2. Configure your settings in `configs/`.

3. See `notebooks/BasicsOfIRT.ipynb` for an introduction to the IRT models used (1PL/2PL/3PL/4PL).

## Usage

### 1. Run LLM Evaluations

`scripts/benchmarks/evaluate_model_gsm.sh` and `evaluate_model_mmlu.sh` are SLURM job scripts that run a list of HuggingFace models through `lm-evaluation-harness` with `vllm` as the backend, writing results to `results/gsm/` and `results/mmlu/` respectively (created locally; not tracked in this repo). Set `HF_TOKEN` in your environment before running:

```bash
export HF_TOKEN=hf_your_token_here
sbatch scripts/benchmarks/evaluate_model_gsm.sh
sbatch scripts/benchmarks/evaluate_model_mmlu.sh
```

Adjust the model list, SLURM directives (partition, GPU, memory), and `--output_path` as needed for your own cluster/environment.

### 2. Convert Evaluation Results → IRT Format

The project includes a pipeline to convert raw LLM benchmark evaluation results into the IRT-compatible JSONLINES format required by the IRT-fitting scripts.

#### IRT Format

Each output `.jsonlines` file contains one JSON object per line, one per model (subject):

```json
{"subject_id": "model_name", "responses": {"item_0": 1, "item_1": 0, "item_2": 1, ...}}
```

- `subject_id`: the model directory name (e.g. `braindao__Qwen2.5-14B-Instruct`)
- `responses`: a dictionary mapping `item_{doc_id}` to the binary accuracy score (0 or 1)

#### Supported Datasets

| Dataset | Input directory | Input filename pattern | Accuracy field | Output files |
|---------|----------------|----------------------|----------------|-------------|
| **MMLU** (Global-MMLU) | `results/mmlu/{model}/` | `samples_global_mmlu_{lang}_{domain}_{timestamp}.jsonl` | `acc` | `mmlu_{lang}_{domain}.jsonlines` |
| **MGSM** | `results/gsm/{model}/` | `samples_mgsm_direct_{lang}_{timestamp}.jsonl` | `exact_match` | `mgsm_{lang}.jsonlines` |

#### Quick Start — Run All Conversions

```bash
bash scripts/irt/run_convert_to_irt.sh
```

This converts both MMLU and MGSM results and writes output to `data/irt/` (created locally). For example:

```
data/irt/
├── mmlu_en_business.jsonlines      # MMLU English, business domain
├── mmlu_de_stem.jsonlines          # MMLU German, STEM domain
├── mmlu_zh_humanities.jsonlines    # MMLU Chinese, humanities domain
├── ...
├── mgsm_en.jsonlines               # MGSM English
├── mgsm_de.jsonlines               # MGSM German
├── mgsm_zh.jsonlines               # MGSM Chinese
└── ...
```

#### Python Script — Fine-Grained Control

```bash
# Convert only MMLU
python scripts/irt/convert_to_irt.py --dataset mmlu --results_dir results/mmlu --output_dir data/irt

# Convert only MGSM
python scripts/irt/convert_to_irt.py --dataset mgsm --results_dir results/gsm --output_dir data/irt
```

| Argument | Description |
|----------|-------------|
| `--dataset` | `mmlu` or `mgsm` |
| `--results_dir` | Path to the directory containing model subdirectories with sample JSONL files |
| `--output_dir` | Path to write the IRT-formatted output files |

#### Pipeline Details

1. The script scans all model subdirectories under `--results_dir`
2. For each model, it finds all `samples_*.jsonl` files and parses the language/domain from the filename
3. Each sample record's accuracy (`acc` for MMLU, `exact_match` for MGSM) is converted to a binary 0/1
4. Results are grouped by (language, domain) for MMLU or by language for MGSM
5. For each group, all models are collected into a single `.jsonlines` file where each line represents one model's responses across all items

### 3. Fit IRT Models

```bash
# 1PL via py-irt / SVI
python scripts/irt/irt_onepl.py --data_dir data/irt --output_dir results/1PL

# 2PL / 3PL via MAP (used for the paper's reported results)
python scripts/irt/irt_map.py --model 2pl --data_dir data/irt --output_dir results/2PL_map
python scripts/irt/irt_map.py --model 3pl --data_dir data/irt --output_dir results/3PL_map
```

### 4. Run the Analyses

Scripts under `scripts/core_analysis/` consume the fitted IRT parameters (`results/1PL/`, `results/2PL_map/`, `results/3PL_map/`) and reproduce the cross-lingual concordance, subset-fidelity, and DIF analyses, writing CSVs/figures to `results/core_analysis/`. Each script can be run independently, e.g.:

```bash
python scripts/core_analysis/cross_model_concordance.py
python scripts/core_analysis/dif_anchor.py
python scripts/core_analysis/subset_ranking_fidelity_map.py
```

## Contributing

(To be added)

## License

See LICENSE file for details.
