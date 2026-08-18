# Final demonstration guide

Use this outline for a 5-10 minute individual demonstration. Explain the work in your own words and use the notebook and generated outputs as evidence.

## 1. Problem and data (about 1 minute)

- State the question: can synthetic cardiac-style predictors classify an elevated simulated risk pattern?
- Clarify that every record is synthetic, `cardiac_risk` is not a diagnosis, and the project is educational only.
- Show the data dictionary and identify the target (`cardiac_risk`) and excluded ID (`patient_id`).

## 2. Data preparation and EDA (about 2 minutes)

- Show the quality audit in notebook section 3.
- Explain that exact duplicates are removed, implausible values and invalid categories become missing, and imputation is learned inside the training pipeline to avoid leakage.
- Show the EDA overview and correlation plot. Describe patterns as associations in this synthetic sample, not causal or clinical findings.

## 3. Supervised learning (about 2 minutes)

- Explain the stratified 80/20 train/test split and the two predictor-only engineered features.
- Introduce Logistic Regression as the baseline and Random Forest as the comparison model.
- Show `outputs/model_comparison.csv`. Both models use the same preprocessing and five-fold stratified cross-validation.

## 4. Evaluation (about 1-2 minutes)

- Explain the held-out metrics in `outputs/final_test_results.csv`: accuracy, precision, recall, F1, and ROC-AUC.
- Show the confusion matrix and ROC curve.
- Define false positive as predicting an elevated simulated pattern when the label is lower, and false negative as predicting a lower pattern when the label is elevated.
- Explain that model selection used cross-validated ROC-AUC, not the test set.

## 5. Unsupervised exploration and limitations (about 1-2 minutes)

- Show the PCA/K-means plot and cluster summary. The groups are exploratory descriptions, not diagnoses.
- State the main limitations: synthetic and simplified data, no external validation, no calibration or fairness study, and PCA/K-means assumptions.

## Likely questions

| Question | Concise answer |
| --- | --- |
| Why use a pipeline? | It keeps preprocessing and modeling together and prevents data leakage during cross-validation. |
| Why stratify the split? | It preserves the target-class proportion in training and test data. |
| Why ROC-AUC? | It evaluates how well the classifier ranks the two simulated classes across thresholds. |
| Can this model be used medically? | No. The data and target are synthetic and the project is strictly educational. |
| Why include imperfect values and duplicates? | They provide reproducible evidence for the data-cleaning workflow. |
