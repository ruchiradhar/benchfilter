#!/usr/bin/env python3
"""Convert evaluation results (MMLU / MGSM) into IRT-format JSONLINES files.

Usage:
    python scripts/convert_to_irt.py --dataset mmlu --results_dir results/mmlu --output_dir data/irt
    python scripts/convert_to_irt.py --dataset mgsm --results_dir results/gsm  --output_dir data/irt
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Filename parsers – extract (lang, domain) or (lang,) from sample filenames
# ---------------------------------------------------------------------------

# MMLU pattern: samples_global_mmlu_{lang}_{domain}_{timestamp}.jsonl
# domain can contain underscores (e.g. "social_sciences")
MMLU_PATTERN = re.compile(
    r"^samples_global_mmlu_(?P<lang>[a-z]{2})_(?P<domain>.+?)_\d{4}-\d{2}-\d{2}T.+\.jsonl$"
)

# MGSM pattern: samples_mgsm_direct_{lang}_{timestamp}.jsonl
# Watch out for the special case: samples_mgsm_direct_es_spanish_bench_{timestamp}.jsonl
MGSM_PATTERN = re.compile(
    r"^samples_mgsm_direct_(?P<lang_and_rest>.+?)_\d{4}-\d{2}-\d{2}T.+\.jsonl$"
)


def parse_mmlu_filename(filename: str):
    """Return (lang, domain) or None."""
    m = MMLU_PATTERN.match(filename)
    if m:
        return m.group("lang"), m.group("domain")
    return None


def parse_mgsm_filename(filename: str):
    """Return lang_key (e.g. 'en', 'es_spanish_bench') or None."""
    m = MGSM_PATTERN.match(filename)
    if m:
        return m.group("lang_and_rest")
    return None


# ---------------------------------------------------------------------------
# Read one samples file -> dict  { doc_id: 0|1 }
# ---------------------------------------------------------------------------

def read_samples(filepath: str, acc_key: str) -> dict:
    """Read a JSONL samples file and return {doc_id: accuracy}."""
    responses = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record["doc_id"]
            score = int(record[acc_key])  # 0.0/1.0 -> 0/1
            responses[doc_id] = score
    return responses


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_mmlu(results_dir: str, output_dir: str):
    """Convert MMLU results to IRT format.
    
    Produces one file per (lang, domain): mmlu_{lang}_{domain}.jsonlines
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect: { (lang, domain): [ (model_name, {doc_id: score}) ] }
    collection = defaultdict(list)

    model_dirs = sorted([d for d in results_path.iterdir() if d.is_dir()])
    for model_dir in model_dirs:
        model_name = model_dir.name
        for f in sorted(model_dir.iterdir()):
            if not f.name.endswith(".jsonl"):
                continue
            parsed = parse_mmlu_filename(f.name)
            if parsed is None:
                continue
            lang, domain = parsed
            responses = read_samples(str(f), acc_key="acc")
            collection[(lang, domain)].append((model_name, responses))

    # Write IRT files
    for (lang, domain), model_responses in sorted(collection.items()):
        out_file = output_path / f"mmlu_{lang}_{domain}.jsonlines"
        with open(out_file, "w", encoding="utf-8") as fout:
            for model_name, responses in model_responses:
                # Sort items by doc_id numerically
                sorted_items = sorted(responses.items(), key=lambda x: x[0])
                irt_responses = {f"item_{did}": score for did, score in sorted_items}
                record = {
                    "subject_id": model_name,
                    "responses": irt_responses,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Written: {out_file}  ({len(model_responses)} models, "
              f"{len(model_responses[0][1]) if model_responses else 0} items)")


def convert_mgsm(results_dir: str, output_dir: str):
    """Convert MGSM results to IRT format.
    
    Produces one file per lang: mgsm_{lang}.jsonlines
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect: { lang_key: [ (model_name, {doc_id: score}) ] }
    collection = defaultdict(list)

    model_dirs = sorted([d for d in results_path.iterdir() if d.is_dir()])
    for model_dir in model_dirs:
        model_name = model_dir.name
        for f in sorted(model_dir.iterdir()):
            if not f.name.endswith(".jsonl"):
                continue
            lang_key = parse_mgsm_filename(f.name)
            if lang_key is None:
                continue
            responses = read_samples(str(f), acc_key="exact_match")
            collection[lang_key].append((model_name, responses))

    # Write IRT files
    for lang_key, model_responses in sorted(collection.items()):
        out_file = output_path / f"mgsm_{lang_key}.jsonlines"
        with open(out_file, "w", encoding="utf-8") as fout:
            for model_name, responses in model_responses:
                sorted_items = sorted(responses.items(), key=lambda x: x[0])
                irt_responses = {f"item_{did}": score for did, score in sorted_items}
                record = {
                    "subject_id": model_name,
                    "responses": irt_responses,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Written: {out_file}  ({len(model_responses)} models, "
              f"{len(model_responses[0][1]) if model_responses else 0} items)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert evaluation results to IRT JSONLINES format."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mmlu", "mgsm"],
        help="Dataset type: mmlu or mgsm",
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Path to the results directory (e.g. results/mmlu or results/gsm)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for IRT files",
    )
    args = parser.parse_args()

    print(f"Converting {args.dataset} results from {args.results_dir} ...")

    if args.dataset == "mmlu":
        convert_mmlu(args.results_dir, args.output_dir)
    elif args.dataset == "mgsm":
        convert_mgsm(args.results_dir, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
