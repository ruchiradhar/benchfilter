#!/usr/bin/env bash
# Run IRT format conversion for all datasets.
#
# Usage:
#   bash scripts/irt/run_convert_to_irt.sh
#
# Output goes to data/irt/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_to_irt.py"
OUTPUT_DIR="$PROJECT_DIR/data/irt"

echo "========================================"
echo "  Converting results to IRT format"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================"

# --- MMLU ---
MMLU_DIR="$PROJECT_DIR/results/mmlu"
if [ -d "$MMLU_DIR" ]; then
    echo ""
    echo "[MMLU] Converting from $MMLU_DIR ..."
    python3 "$CONVERT_SCRIPT" \
        --dataset mmlu \
        --results_dir "$MMLU_DIR" \
        --output_dir "$OUTPUT_DIR"
else
    echo "[MMLU] Directory not found: $MMLU_DIR — skipping."
fi

# --- MGSM ---
MGSM_DIR="$PROJECT_DIR/results/gsm"
if [ -d "$MGSM_DIR" ]; then
    echo ""
    echo "[MGSM] Converting from $MGSM_DIR ..."
    python3 "$CONVERT_SCRIPT" \
        --dataset mgsm \
        --results_dir "$MGSM_DIR" \
        --output_dir "$OUTPUT_DIR"
else
    echo "[MGSM] Directory not found: $MGSM_DIR — skipping."
fi

echo ""
echo "========================================"
echo "  All conversions finished."
echo "  Output files:"
ls -1 "$OUTPUT_DIR"/*.jsonlines 2>/dev/null || echo "  (none)"
echo "========================================"
