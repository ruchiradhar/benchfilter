#!/usr/bin/env bash
# Submit/run 1PL IRT training from converted files in data/irt.
#
# Usage:
#   bash scripts/irt/run_irt_onepl.sh <dataset> <language> [category]
#
# Examples:
#   bash scripts/irt/run_irt_onepl.sh mgsm en
#   bash scripts/irt/run_irt_onepl.sh mmlu en humanities
#
# Slurm examples:
#   sbatch scripts/irt/run_irt_onepl.sh mgsm en
#   sbatch -p gpu --mem=32G --time=12:00:00 scripts/irt/run_irt_onepl.sh mmlu en stem

set -euo pipefail

# Initialize cluster environment
if [[ -f /etc/profile.d/modules.sh ]]; then
    . /etc/profile.d/modules.sh
fi

if type module >/dev/null 2>&1; then
    module load anaconda3/5.3.1
    module load cuda/11.8
fi

if command -v conda >/dev/null 2>&1; then
    export PS1="${PS1-}"
    eval "$(conda shell.bash hook)"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "benchfilterenv" ]]; then
        conda activate benchfilterenv
    fi
else
    echo "conda was not found in PATH. Cannot activate benchfilterenv." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || { cd "$SCRIPT_DIR/../.." && pwd; })}"
PY_SCRIPT="$PROJECT_DIR/scripts/irt/irt_onepl.py"
DATA_DIR="$PROJECT_DIR/data/irt"
OUTPUT_DIR="$PROJECT_DIR/results/1PL"
LOG_DIR="$PROJECT_DIR/logs/irt"

DATASET="${1:-}"
LANGUAGE="${2:-}"
CATEGORY="${3:-}"

if [[ -z "$DATASET" || -z "$LANGUAGE" ]]; then
    echo "Usage: bash scripts/irt/run_irt_onepl.sh <dataset> <language> [category]" >&2
    echo "  dataset: mgsm | mmlu" >&2
    echo "  language: e.g., en, de, es, zh" >&2
    echo "  category: optional for mmlu only" >&2
    exit 1
fi

if [[ "$DATASET" != "mgsm" && "$DATASET" != "mmlu" ]]; then
    echo "Invalid dataset '$DATASET'. Use 'mgsm' or 'mmlu'." >&2
    exit 1
fi

if [[ "$DATASET" == "mgsm" && -n "$CATEGORY" ]]; then
    echo "CATEGORY is only supported for dataset=mmlu." >&2
    exit 1
fi

EPOCHS="${EPOCHS:-2000}"
LOG_EVERY="${LOG_EVERY:-500}"
DEVICE="${DEVICE:-cpu}"
DROPOUT="${DROPOUT:-0.5}"
LR="${LR:-0.1}"
LR_DECAY="${LR_DECAY:-0.9999}"
SEED="${SEED:-}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "========================================"
echo "  Running 1PL IRT"
echo "  Project dir : $PROJECT_DIR"
echo "  Data dir    : $DATA_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Dataset     : $DATASET"
echo "  Language    : $LANGUAGE"
if [[ -n "$CATEGORY" ]]; then
    echo "  Category    : $CATEGORY"
fi
echo "  Device      : $DEVICE"
echo "========================================"

CMD=(
    python3 "$PY_SCRIPT"
    --dataset "$DATASET"
    --language "$LANGUAGE"
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --log_dir "$LOG_DIR"
    --epochs "$EPOCHS"
    --log_every "$LOG_EVERY"
    --dropout "$DROPOUT"
    --lr "$LR"
    --lr_decay "$LR_DECAY"
    --device "$DEVICE"
)

if [[ -n "$CATEGORY" ]]; then
    CMD+=(--category "$CATEGORY")
fi

if [[ -n "$SEED" ]]; then
    CMD+=(--seed "$SEED")
fi

"${CMD[@]}"

echo ""
echo "========================================"
echo "  Finished 1PL IRT"
echo "  Results in: $OUTPUT_DIR"
echo "========================================"
