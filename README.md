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

(To be added as the project develops)

## Contributing

(To be added)

## License

See LICENSE file for details.
