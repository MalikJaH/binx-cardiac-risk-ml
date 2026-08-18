# Cardiac Patient Monitoring — Machine Learning Analysis

This individual project implements a complete curriculum-aligned machine-learning workflow on a deterministic synthetic cardiac dataset. It cleans and explores the data, compares two supervised classifiers, evaluates them with stratified cross-validation and a held-out test set, builds reusable Scikit-learn pipelines, and explores patient-like groups with PCA and K-means clustering.

> **Educational use only:** the records and target are synthetic. This project does not diagnose disease, recommend treatment, provide emergency guidance, or represent a clinically validated system.

## Project question

Can routinely shaped synthetic cardiac attributes classify an **elevated simulated cardiac-risk pattern** (`cardiac_risk = 1`) versus a lower pattern (`cardiac_risk = 0`)? The target is generated from probabilistic relationships among synthetic predictors and is not a medical diagnosis.

## Repository structure

```text
data/
  cardio_patients_raw.csv       generated synthetic observations
  data_dictionary.md            definitions and valid ranges
notebooks/
  cardiac_patient_monitoring.ipynb
src/
  generate_data.py              deterministic data generator (seed 42)
models/
  cardiac_risk_pipeline.joblib  selected fitted pipeline after execution
outputs/
  *.png / *.csv                 charts and result summaries
docs/
  milestones.md                 evidence for all seven review gates
DEMO_GUIDE.md                   5–10 minute presentation outline
requirements.txt               exact Python dependencies
```

## Methods

- Pandas/NumPy validation of duplicates, missing values, categories, and physiological plausibility ranges.
- Descriptive statistics, conditional proportions, distributions, class balance, outlier plots, correlations, and relationship plots using Matplotlib.
- Predictor-only feature engineering: age-adjusted maximum heart-rate ratio and age–blood-pressure interaction.
- Stratified 80/20 train/test split with random seed 42.
- Logistic Regression baseline and Random Forest comparison, each in an identical preprocessing Pipeline.
- Five-fold stratified cross-validation using accuracy, precision, recall, F1, and ROC-AUC.
- Held-out confusion matrix, classification report, ROC curve, and plain-language error interpretation.
- Median/mode imputation, standardization, and one-hot encoding learned inside each pipeline to avoid leakage.
- Standardized PCA visualization and three-cluster K-means exploratory analysis with a silhouette score.

## Run from a clean environment

Python 3.12 is recommended. From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/generate_data.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/cardiac_patient_monitoring.ipynb --ExecutePreprocessor.timeout=300
```

The last command executes the notebook from top to bottom in place and creates the files in `outputs/` and `models/`. To explore interactively instead:

```powershell
python -m jupyter lab notebooks/cardiac_patient_monitoring.ipynb
```

On macOS/Linux, replace the activation command with `source .venv/bin/activate`; the remaining commands are the same.

## Reproduce the data

The raw CSV is committed for convenience. It can be recreated exactly at any time:

```powershell
python src/generate_data.py
```

The generator uses seed 42 and intentionally adds a small number of duplicates, missing values, invalid values, and one invalid category. These issues are teaching evidence for the notebook's cleaning audit.

## Outputs and interpretation

After successful execution, inspect:

- `outputs/model_comparison.csv` for cross-validation and held-out metrics.
- `outputs/confusion_matrix.png` and `outputs/roc_curve.png` for classification evaluation.
- `outputs/pca_clusters.png` and `outputs/cluster_summary.csv` for exploratory grouping.
- `models/cardiac_risk_pipeline.joblib` for the selected fitted end-to-end preprocessing/model pipeline.

Exact scores are generated rather than hard-coded so they remain traceable to the executed environment. Model selection uses mean cross-validated ROC-AUC; the test set is used once for final reporting, not model selection.

In the verified seed-42 run, Logistic Regression was selected with mean five-fold ROC-AUC **0.828 +/- 0.027**. On the 240-row held-out test set it achieved **0.746 accuracy, 0.742 precision, 0.760 recall, 0.751 F1, and 0.828 ROC-AUC**. The exploratory three-cluster solution had a modest silhouette score of **0.223**, so the groups should be treated as descriptive rather than strongly separated. Exact values are recorded in `outputs/final_test_results.csv` and `outputs/model_comparison.csv`.

## Limitations

- The dataset and target mechanism are synthetic and simplified.
- Associations deliberately built into the generator can make performance look more stable than on real data.
- Important clinical, temporal, social, medication, and measurement context is absent.
- K-means assumes geometric groups and PCA discards information when projecting to two dimensions.
- No external validation, calibration study, fairness audit, prospective evaluation, or clinical review was performed.
- The saved model is an educational artifact and must not be used for patient care.
