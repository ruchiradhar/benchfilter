# BenchFilter

A research project for evaluating and comparing different Item Response Theory (IRT) methods to identify the most relevant subset of items/questions based on capability assessment using Large Language Models.

## Project Structure

```
benchfilter/
├── README.md                 # This file
├── LICENSE                   # License information
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── configs/                 # Configuration files
│   ├── config._train.yaml   # Training configuration
│   └── ...                  # Other config files
│
├── data/                    # Data storage
│   ├── raw/                 # Original, unprocessed data
│   └── processed/           # Cleaned and preprocessed data
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 
│   ├── 
│   └── 
│
├── src/                     # Source code
│   ├── irt_methods/         # IRT method implementations
│   ├── llm_interface/       # LLM integration code
│   ├── evaluation/          # Evaluation metrics and methods
│   └── utils/               # Utility functions
│
├── scripts/                 # Standalone scripts
│   ├── 
│   └── 
│
├── results/                 # Experiment results
│   ├── comparisons/         # Method comparison results
│   └── plots/               # Generated visualizations
│

```

## Overview

This project aims to:
1. Run Benchmarks with different LLMs to record response patterns
2. Implement various IRT methods for Benchmarks to find parameters
3. Compare sampling methods to find the most effective subset for a capability

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your settings in `configs/`

3. Start with the notebooks in `notebooks/` for exploration

## Usage

### Data Conversion: Evaluation Results → IRT Format

The project includes a pipeline to convert raw LLM benchmark evaluation results (from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)) into the IRT-compatible JSONLINES format required by the IRT methods.

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
bash scripts/run_convert_to_irt.sh
```

This converts both MMLU and MGSM results and writes output to `data/irt/`. For example:

```
data/irt/
├── mmlu_en_business.jsonlines      # MMLU English, business domain (70 models × 58 items)
├── mmlu_de_stem.jsonlines          # MMLU German, STEM domain
├── mmlu_zh_humanities.jsonlines    # MMLU Chinese, humanities domain
├── ...
├── mgsm_en.jsonlines              # MGSM English (22 models × 250 items)
├── mgsm_de.jsonlines              # MGSM German
├── mgsm_zh.jsonlines              # MGSM Chinese
└── ...
```

#### Python Script — Fine-Grained Control

```bash
# Convert only MMLU
python scripts/convert_to_irt.py --dataset mmlu --results_dir results/mmlu --output_dir data/irt

# Convert only MGSM
python scripts/convert_to_irt.py --dataset mgsm --results_dir results/gsm --output_dir data/irt
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

## Contributing

(To be added)

## License

See LICENSE file for details.
