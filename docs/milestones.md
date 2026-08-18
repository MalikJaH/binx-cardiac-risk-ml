# Project milestone evidence

This file maps the retained project artifacts to the seven curriculum review gates. It records evidence available in this repository; it does not claim that any particular item was completed on a specific calendar day.

| Milestone | Required evidence | Retained project evidence |
| --- | --- | --- |
| M1: Environment and data | Reproducible environment, loaded data, initial cleaning | `requirements.txt`, `src/generate_data.py`, `data/cardio_patients_raw.csv`, `data/data_dictionary.md`, and notebook sections 1-3. |
| M2: EDA and statistics | Descriptive statistics, data-quality review, visualizations | Notebook section 4 and `outputs/eda_overview.png`, `outputs/correlation_matrix.png`. |
| M3: Baseline supervised model | Defined classification task, train/test split, baseline metrics | Notebook sections 5-6; Logistic Regression is the baseline. |
| M4: Comparison and evaluation | Second classifier, cross-validation, classification metrics, confusion matrix | Notebook sections 7-8, `outputs/model_comparison.csv`, `outputs/final_test_results.csv`, `outputs/confusion_matrix.png`, and `outputs/roc_curve.png`. |
| M5: Feature engineering and pipeline | Predictor-only features and repeatable preprocessing/model workflow | Notebook sections 5-6 and `models/cardiac_risk_pipeline.joblib`. |
| M6: Unsupervised analysis | Clustering/PCA visualization and interpretation | Notebook section 9, `outputs/pca_clusters.png`, `outputs/cluster_summary.csv`, `outputs/pca_dbscan_clusters.png`, and `outputs/dbscan_cluster_summary.csv`. |
| M7: Final documentation and demonstration | Clear instructions, result summary, limitations, and demo preparation | `README.md`, this file, `DEMO_GUIDE.md`, executed notebook, and generated outputs. |

## Reproducibility check

Run the commands in `README.md` from the repository root. The generator uses seed 42, and the notebook recreates the charts, CSV summaries, and saved pipeline. The data is synthetic and the `cardiac_risk` label is an educational simulated pattern, not a clinical diagnosis.
