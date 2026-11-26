# Network Intrusion Detection System (NIDS)

**Author:** Lokesh Deshwal  
**Roll No.:** 2401421062

A clean, modular machine learning pipeline for detecting malicious network traffic on the **UNSW-NB15** dataset, featuring automated preprocessing, multi-model benchmarking, risk-tiered inference, and CI testing.

## Project Structure

```text
Network_Intrusion_System/
├── .github/workflows/ci.yaml   # GitHub Actions CI (pytest)
├── dataset/                    # Place UNSW-NB15 .parquet or .csv files here
├── notebooks/                  # Lightweight EDA & model comparison notebooks
│   ├── 01_eda.ipynb
│   └── 02_model_comparison.ipynb
├── src/nids/                   # Modular Python package
│   ├── __init__.py
│   ├── cli.py                  # CLI commands (demo, train, predict, convert)
│   ├── config.py               # Centralized paths, target, and model settings
│   ├── data.py                 # Parquet/CSV loading, conversion, synthetic demo data
│   ├── features.py             # ColumnTransformer preprocessing pipelines
│   ├── predict.py              # Inference & risk-tier scoring (LOW/MEDIUM/HIGH)
│   └── train.py                # Multi-model training, evaluation, and artifact export
├── tests/                      # Automated unit and smoke tests
│   ├── test_pipeline.py
│   └── test_smoke.py
├── main.py                     # CLI entrypoint
├── pyproject.toml              # Package & pytest configuration
└── requirements.txt            # Python dependencies
```

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Instant Demo (No Dataset Download Required)

Generate a synthetic UNSW-NB15-like dataset, train & compare models, save the best pipeline to `artifacts/`, and score sample traffic:

```bash
python main.py demo
```

### 3. Train on Full UNSW-NB15 Data

Place `train_full.parquet` and `test_full.parquet` inside `dataset/` (or convert CSVs using `python main.py convert`), then run:

```bash
python main.py train --train-path dataset/train_full.parquet --test-path dataset/test_full.parquet
```

### 4. Score New Traffic

```bash
python main.py predict --input dataset/test_full.parquet --output artifacts/predictions.csv
```

## Key Improvements Over Notebook-Only Workflows

- **Modular `src/nids` Package**: Separates configuration, data I/O, preprocessing, training, and inference from notebooks.
- **Risk-Tiered Scoring**: Outputs both intrusion probability and human-readable alert tiers (`LOW`, `MEDIUM`, `HIGH`).
- **Offline Demo Mode**: Built-in synthetic traffic generator lets you test and verify the full pipeline without large dataset files.
- **Automated Testing**: `pytest` suite validates imports, data conversion, training, and inference end-to-end.

## Running Tests

```bash
pytest -v
```
