# download

This directory contains the code and requirements for downloading the training and test datasets used for FathomNetCLEF 2026.

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt` (available via PyPI)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python download.py [-h] [-o OUTPUT_DIR] [--min-workers MIN_WORKERS] [--max-workers MAX_WORKERS] [--initial-workers INITIAL_WORKERS] dataset_path
```

The script will autoscale the number of workers based on server failures.
