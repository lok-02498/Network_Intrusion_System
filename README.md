# Network Intrusion Detection System

**Author:** Lokesh Deshwal  
**Roll No.:** 2401421062

Machine learning and deep learning experiments on the **UNSW-NB15** network intrusion dataset, including exploratory data analysis, baseline modeling, XGBoost tuning, and ML vs. DL comparisons.

## Project Structure

```text
.
├── .github/workflows/ci.yaml
├── .gitignore
├── capstone/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_best_model_tuning.ipynb
│   └── 04_ml_vs_dl.ipynb
├── dataset/
│   ├── test_full.parquet
│   └── train_full.parquet
├── notebooks/
│   ├── 01_Data Audit & EDA.ipynb
│   ├── 02_Baseline Model.ipynb
│   └── 03_XGBoost Model.ipynb
├── tests/
│   └── test_smoke.py
├── praquet_converter.py
├── README.md
└── requirements.txt
```

## Setup & Usage

```bash
pip install -r requirements.txt
pytest -v
```

To convert raw UNSW-NB15 CSV files into Parquet:

```bash
python praquet_converter.py --dataset-dir dataset
```
