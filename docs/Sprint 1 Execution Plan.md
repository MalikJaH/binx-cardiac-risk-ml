#  Sprint 1 Execution Plan

**Project:** Cardiac Patient Monitoring — Synthetic Cardiac-Risk Classification
**Phase:** Phase 3 Capstone
**Sprint:** Sprint 1 of 4
**Week:** Week 6
**Duration:** 5 working days — 40 hours
**Workflow:** Jupyter Notebook + Python + scikit-learn + TensorFlow/Keras + GitHub
**Primary Sprint Outcome:** A validated dataset, documented EDA, reproducible Logistic Regression baseline, and a benchmark that later models must beat.
**Status:** Ready for execution

---

# 1. Sprint 1 Goal

> **Understand and validate the synthetic cardiac dataset, establish a leakage-safe Logistic Regression baseline, and use that baseline as the benchmark for all later capstone models.**

Sprint 1 will also introduce a first TensorFlow/Keras neural network as part of the Week 6 deep-learning learning objectives.

The neural network is **not required to beat the baseline during Sprint 1**. Its purpose this week is to demonstrate the complete deep-learning workflow and establish the first comparison point.

---

# 2. Dataset Locked for Sprint 1

Sprint 1 will use the generated synthetic cardiac dataset:

```text
synthetic_cardiac_100k.csv
```

Dataset characteristics:

* 100,000 rows
* 16 columns
* Binary target: `cardiac_risk`
* Approximately 30% positive class
* Synthetic records only
* Intentional missing values
* Intentional exact duplicate rows
* Intentional invalid/outlier values for validation practice
* No real patient or clinical data

## Target
cardiac_risk
```

Where:

```text
0 = lower simulated cardiac-risk pattern
1 = elevated simulated cardiac-risk pattern
```

---


---

# 3. Primary Evaluation Metric

## Model-selection metric

```text
ROC-AUC
```

The official benchmark will be:

```text
Mean 5-Fold CV ROC-AUC ± Standard Deviation
```

Additional metrics:

```text
Accuracy
Precision
Recall
F1-score
```

ROC-AUC is the metric that future models must primarily beat.

For example, if Sprint 1 produces:

```text
Logistic Regression
Mean CV ROC-AUC = 0.81
```

then the benchmark statement becomes:

> Future candidate models should aim to outperform a cross-validated ROC-AUC of 0.81.

The actual number must come from the executed notebook. It must not be written in advance.

---

# 5. Repository Structure

By the end of Sprint 1, the repository should contain:

```text
cardiac-risk-capstone/
│
├── data/
│   ├── raw/
│   │   └── synthetic_cardiac_100k.csv
│   │
│   └── data_dictionary.md
│
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_logistic_regression_baseline.ipynb
│   └── 03_neural_network_intro.ipynb
│
├── src/
│   ├── __init__.py
│   └── validation.py
│
├── reports/
│   ├── baseline_cv_metrics.csv
│   ├── sprint1_findings.md
│   └── sprint1_retrospective.md
│
├── models/
│
├── README.md
├── requirements.txt
└── .gitignore
```

No final neural-network model needs to be saved during Sprint 1 unless it becomes useful later.

---

# 6. Sprint 1 Backlog

| ID    | Backlog Item                                                              | Priority                 |
| ----- | ------------------------------------------------------------------------- | ------------------------ |
| S1-01 | Sprint kickoff, repository setup, branch creation, and scope confirmation | Must                     |
| S1-02 | Dataset audit and validation rules                                        | Must                     |
| S1-03 | Exploratory data analysis                                                 | Must                     |
| S1-04 | Development/test split and preprocessing pipeline                         | Must                     |
| S1-05 | Logistic Regression baseline with stratified 5-fold CV                    | Must                     |
| S1-06 | Record baseline benchmark and supporting metrics                          | Must                     |
| S1-07 | Build first TensorFlow/Keras neural network                               | Must for Week 6 learning |
| S1-08 | Mentor notebook/code review and corrections                               | Must                     |
| S1-09 | Sprint Review, documentation, and retrospective                           | Must                     |

---

# 7. Day 1 — Sprint Kickoff, Dataset Audit & Baseline First

**Goal:** Finish the core Sprint 1 requirement as early as possible: understand the dataset and establish the first reproducible baseline.

## 7.1 Sprint Planning

Confirm:

* Sprint goal
* dataset
* target
* primary metric
* modelling restrictions
* Sprint Definition of Done

Create the branch:

```bash
git checkout main
git pull
git checkout -b feature/sprint-1-baseline
```

---

## 7.2 Dataset Audit

Create:

```text
notebooks/01_data_audit_eda.ipynb
```

Start with:

```python
import pandas as pd

df = pd.read_csv("data/raw/synthetic_cardiac_100k.csv")

df.shape
df.head()
df.info()
df.describe(include="all").T
```

Audit:

* row count
* column count
* data types
* duplicate rows
* missing values
* invalid numeric values
* invalid categories
* target balance
* unique values
* identifier quality

### Validation rules

Use the following expected clean ranges:

```text
age:             18–90
resting_bp:      85–210
cholesterol:     100–430
bmi:             16–48
max_heart_rate:  65–205
```

Binary columns must contain only:

```text
0
1
```

Allowed categorical values must be checked against the data dictionary.

---

## 7.3 Duplicate Handling

Count exact duplicates:

```python
df.duplicated().sum()
```

Document the result.

Remove exact duplicates before the modelling split.

Do not silently remove them without recording how many were found.

---

## 7.4 Invalid Values

Invalid values should be identified using the documented rules.

Instead of inventing replacements, convert invalid numeric measurements to missing values:

```python
df.loc[invalid_condition, column] = pd.NA
```

Their eventual imputation will occur inside the modelling pipeline.

This keeps validation logic separate from learned preprocessing.

---

## 7.5 Begin EDA

At minimum:

* target class distribution
* numeric distributions
* missing-value summary
* categorical distributions
* cardiac-risk rate by important predictors

Charts should have:

* titles
* labelled axes
* readable legends
* short Markdown interpretation

---

## 7.6 Create the Development/Test Split

After deterministic validation and duplicate removal:

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["patient_id", "cardiac_risk"])
y = df["cardiac_risk"]

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

After this point:

> **Do not inspect test performance during Sprint 1.**

---

## 7.7 Build the First Baseline

Create:

```text
notebooks/02_logistic_regression_baseline.ipynb
```

Build a preprocessing pipeline.

### Numeric pipeline

```text
Median imputation
→
StandardScaler
```

### Categorical pipeline

```text
Most-frequent imputation
→
OneHotEncoder(handle_unknown="ignore")
```

Then:

```text
ColumnTransformer
→
LogisticRegression
```

Use a transparent baseline without hyperparameter tuning.

Example:

```python
LogisticRegression(
    max_iter=2000
)
```

Evaluate it using:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Record:

* ROC-AUC
* accuracy
* precision
* recall
* F1

The most important result is:

```text
mean_cv_roc_auc
```

---

## Day 1 Definition of Done

* [ ] Sprint branch exists.
* [ ] Dataset is loaded successfully.
* [ ] Dataset quality audit is documented.
* [ ] Duplicate count is recorded.
* [ ] Missing values are recorded.
* [ ] Invalid values are identified.
* [ ] Target balance is recorded.
* [ ] Development/test split exists.
* [ ] Test set has not been used for model selection.
* [ ] Logistic Regression pipeline runs.
* [ ] 5-fold CV metrics are recorded.
* [ ] Baseline ROC-AUC benchmark is written in Markdown.
* [ ] Initial commits are pushed.
* [ ] Draft pull request is opened.

Suggested commits:

```bash
git add .
git commit -m "feat: add Sprint 1 dataset audit and validation"

git add .
git commit -m "feat: add leakage-safe logistic regression baseline"

git push -u origin feature/sprint-1-baseline
```

Open a **Draft Pull Request** at the end of Day 1.

---

# 8. Day 2 — Complete EDA & Deep-Learning Foundations

**Goal:** Complete the analytical understanding of the dataset while learning the concepts needed for the neural network.

## EDA work

Complete:

### Univariate analysis

Numeric distributions:

```text
age
resting_bp
cholesterol
bmi
max_heart_rate
```

Categorical distributions:

```text
sex
chest_pain_type
resting_ecg
smoking
diabetes
exercise_angina
```

### Target analysis

Plot:

```text
cardiac_risk = 0
cardiac_risk = 1
```

Report percentages.

### Relationship analysis

Examples:

```text
Age vs cardiac_risk
BMI vs cardiac_risk
Resting BP vs cardiac_risk
Cholesterol vs cardiac_risk
Diabetes vs cardiac_risk
Smoking vs cardiac_risk
Exercise angina vs cardiac_risk
Chest pain type vs cardiac_risk
```

### Correlation analysis

Calculate numeric correlation.

Do not interpret correlation as causation.

---

## Required EDA Findings

Write at least three meaningful findings in Markdown.

Example structure:

```text
Finding 1:
...

Finding 2:
...

Finding 3:
...
```

Also document:

> Because the dataset is synthetically generated, observed relationships reflect the assumptions and probability structure used during generation and must not be interpreted as clinical evidence.

---

## Deep-Learning Learning Objectives

Study and document:

```text
Neuron
Weights
Bias
Layers
Activation functions
Forward propagation
```

Understand:

```text
output = activation(XW + b)
```

No complicated neural network is required yet.

---

## Day 2 Definition of Done

* [ ] EDA notebook is complete.
* [ ] At least three findings are documented.
* [ ] Class imbalance is discussed.
* [ ] Important feature relationships are visualized.
* [ ] Synthetic-data limitation is clearly stated.
* [ ] Correlation analysis is included.
* [ ] Baseline notebook runs from top to bottom.
* [ ] Work is committed and pushed to the Draft PR.

Suggested commit:

```bash
git commit -am "feat: complete Sprint 1 exploratory data analysis"
git push
```

---

# 9. Day 3 — First Neural Network & Mentor Review

**Goal:** Apply the Week 6 neural-network concepts and prepare a clean project state for mentor review.

Create:

```text
notebooks/03_neural_network_intro.ipynb
```

Use the same development data and compatible preprocessing approach.

A reasonable first architecture:

```python
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
```

Compile with:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name="roc_auc"),
        "accuracy"
    ]
)
```

Use a validation subset from the development data.

Do not use the final test set.

Add:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Record:

* training loss
* validation loss
* training ROC-AUC
* validation ROC-AUC
* accuracy

Plot:

```text
Training vs validation loss
Training vs validation ROC-AUC
```

---

# Mentor Review — Day 3

The Draft PR should contain enough work for meaningful review.

The mentor should be able to inspect:

```text
Dataset audit
EDA
Validation decisions
Train/test strategy
Preprocessing
Logistic Regression baseline
Baseline metrics
Initial neural network
```

Update the PR description with:

```text
## Sprint Goal

## Completed

## Dataset Findings

## Baseline Result

## Validation Performed

## Current Neural Network Result

## Questions / Blockers
```

After review:

* read every comment
* respond to methodology questions
* fix valid issues
* rerun affected notebook cells
* push corrections

Suggested commit:

```bash
git commit -am "fix: address mentor review feedback"
git push
```

---

# 10. Day 4 — Deep-Learning Training Loop & Controlled Experiments

**Goal:** Understand why the neural network behaves as it does instead of randomly changing settings.

Study:

```text
Loss
Forward propagation
Backpropagation
Gradient descent
Learning rate
Optimizers
Epochs
Batch size
Batch normalization
Overfitting
Early stopping
```

Run only a small number of controlled experiments.

For example:

### Experiment A

```text
32 → 16 → 1
ReLU
Adam
```

### Experiment B

```text
64 → 32 → 1
ReLU
Adam
```

### Experiment C

```text
64
BatchNormalization
32
BatchNormalization
1
```

Change one meaningful architectural idea at a time where possible.

Do not run dozens of configurations simply searching for the highest score.

Create a comparison table:

| Model               |    Validation ROC-AUC | Notes               |
| ------------------- | --------------------: | ------------------- |
| Logistic Regression | Official CV benchmark | Baseline            |
| Neural Network A    |                   ... | Simple network      |
| Neural Network B    |                   ... | Increased capacity  |
| Neural Network C    |                   ... | Batch normalization |

The Logistic Regression score remains the official Sprint 1 benchmark.

The neural-network results are exploratory because their validation methodology is not yet the final model-selection procedure.

---

## Day 4 Definition of Done

* [ ] Neural network trains successfully.
* [ ] Loss curve is shown.
* [ ] Validation metric curve is shown.
* [ ] Overfitting is checked.
* [ ] Early stopping is implemented.
* [ ] At least one controlled comparison is documented.
* [ ] Results are compared with the Logistic Regression baseline.
* [ ] No final-test evaluation has occurred.

---

# 11. Day 5 — Sprint Review, Documentation & Retrospective

**Goal:** Close Sprint 1 with reproducible evidence rather than unfinished notebook experiments.

---

## 11.1 Final Reproducibility Check

Restart each notebook kernel.

Run:

```text
Kernel
→ Restart
→ Run All
```

Required notebooks:

```text
01_data_audit_eda.ipynb
02_logistic_regression_baseline.ipynb
03_neural_network_intro.ipynb
```

All must execute without manual hidden state.

---

## 11.2 Save Baseline Metrics

Create:

```text
reports/baseline_cv_metrics.csv
```

Suggested structure:

```text
model,metric,cv_mean,cv_std
logistic_regression,roc_auc,...
logistic_regression,accuracy,...
logistic_regression,precision,...
logistic_regression,recall,...
logistic_regression,f1,...
```

---

## 11.3 Official Benchmark Statement

Add a Markdown section:

```markdown
## Sprint 1 Baseline Benchmark

Logistic Regression achieved a mean 5-fold cross-validated
ROC-AUC of **X.XXX ± X.XXX** on the development set.

This is the official baseline benchmark for the capstone.

Future candidate models should aim to outperform this ROC-AUC
under a comparable validation procedure.

The final test set remains untouched and will be reserved for
the capstone's final unbiased evaluation.
```

Replace `X.XXX` with the actual executed results.

---

# 12. Sprint Review

Demo the following to the mentor:

1. Dataset
2. Data-quality audit
3. Important EDA findings
4. Preprocessing pipeline
5. Logistic Regression baseline
6. Cross-validation result
7. Official benchmark
8. First Keras neural network
9. Training curves
10. Comparison with baseline
11. GitHub PR and project structure

Questions Sprint Review must answer:

```text
What dataset are we using?
What is the target?
What problems were found in the data?
How were they handled?
How was leakage prevented?
What is the baseline?
What is the benchmark ROC-AUC?
Did the neural network improve on it?
What remains for Sprint 2?
```

Incomplete work must move to Sprint 2 with a documented reason.

---

# 13. Sprint Retrospective

Create:

```text
reports/sprint1_retrospective.md
```

Answer:

## What went well?

Examples:

```text
The preprocessing pipeline was reproducible.
The baseline was established early.
The dataset audit identified quality problems before modelling.
```

## What did not go well?

Record actual issues rather than inventing successes.

## What should improve?

Identify one or two concrete improvements.

## Sprint 2 Action

Choose one specific action.

Example:

> Use a consistent experiment-tracking table for every candidate model so model comparisons remain reproducible.

---

# 14. Daily Stand-Up Format

At the beginning of each day, record a short update:

```markdown
### Daily Stand-Up — Day X

**Completed yesterday**
- ...

**Today**
- ...

**Blockers**
- None / ...
```

The stand-up should take approximately three minutes.

---

# 15. Git Workflow

Use one Sprint feature branch:

```text
feature/sprint-1-baseline
```

Recommended commit history:

```text
chore: initialize Sprint 1 project structure

feat: add dataset audit and validation rules

feat: complete cardiac dataset EDA

feat: add leakage-safe logistic regression baseline

feat: add introductory Keras neural network

fix: address mentor review feedback

docs: add Sprint 1 results and retrospective
```

Push regularly:

```bash
git push
```

The Draft PR opened on Day 1 remains open through the sprint.

Day 3:

```text
Mentor Review
```

Day 5:

```text
Final review
→
Approval
→
Merge
```

---

# 16. Sprint 1 Acceptance Criteria

Sprint 1 is complete only when:

## Dataset

* [ ] `synthetic_cardiac_100k.csv` is the documented project dataset.
* [ ] Dataset contains approximately 100,000 records.
* [ ] `patient_id` is excluded from modelling.
* [ ] Target definition is documented.
* [ ] Missing values are audited.
* [ ] Duplicate rows are audited.
* [ ] Invalid values are audited.
* [ ] Invalid categories are checked.
* [ ] Target balance is documented.
* [ ] Data dictionary exists.

## EDA

* [ ] Descriptive statistics are included.
* [ ] Numeric distributions are visualized.
* [ ] Categorical distributions are examined.
* [ ] Class balance is visualized.
* [ ] Relevant feature/target relationships are examined.
* [ ] Correlation analysis is included.
* [ ] At least three findings are documented.
* [ ] Synthetic-data limitations are stated.

## Leakage Prevention

* [ ] Exact duplicates are removed before splitting.
* [ ] Development/test split uses stratification.
* [ ] `random_state=42` is recorded.
* [ ] Learned preprocessing is fitted only through pipelines.
* [ ] Final test data has not been used for feature selection.
* [ ] Final test data has not been used for hyperparameter selection.
* [ ] Final test performance has not been used to choose a model.

## Baseline

* [ ] Logistic Regression is implemented.
* [ ] Numeric missing values are handled inside preprocessing.
* [ ] Numeric scaling is included.
* [ ] Categorical missing values are handled.
* [ ] Categorical variables are encoded safely.
* [ ] 5-fold stratified CV is used.
* [ ] ROC-AUC is the primary metric.
* [ ] Accuracy is recorded.
* [ ] Precision is recorded.
* [ ] Recall is recorded.
* [ ] F1 is recorded.
* [ ] Mean and standard deviation are recorded.
* [ ] Official baseline benchmark is written in Markdown.

## Deep Learning

* [ ] A Keras Sequential model is built.
* [ ] Hidden-layer activation is understood and documented.
* [ ] Sigmoid output is used for binary classification.
* [ ] Binary cross-entropy loss is used.
* [ ] Adam or another justified optimizer is used.
* [ ] Training and validation performance are plotted.
* [ ] Early stopping is used.
* [ ] The result is compared with the baseline.

## Engineering

* [ ] Work exists on `feature/sprint-1-baseline`.
* [ ] Draft PR is opened.
* [ ] Mentor review occurs on Day 3.
* [ ] Review comments are addressed.
* [ ] Notebooks run top-to-bottom.
* [ ] Dependencies are recorded.
* [ ] README is updated.
* [ ] Sprint Review is completed.
* [ ] Sprint Retrospective is completed.
* [ ] PR is approved before merge.

---

# 17. Sprint 1 Final Deliverables

At the end of Week 6, the repository must visibly contain:

```text
✅ 100k synthetic cardiac dataset
✅ Data dictionary
✅ Dataset validation audit
✅ Complete EDA notebook
✅ Three or more EDA findings
✅ Leakage-safe development/test split
✅ Reproducible preprocessing pipeline
✅ Logistic Regression baseline
✅ 5-fold cross-validation results
✅ Official ROC-AUC benchmark
✅ Baseline metrics CSV
✅ First Keras neural network
✅ Neural-network training curves
✅ Baseline vs neural-network discussion
✅ Draft PR
✅ Mentor review evidence
✅ Sprint Review
✅ Sprint Retrospective
```

---

# 18. What Sprint 1 Does Not Do

Sprint 1 will **not**:

* claim medical or clinical validity;
* use real patient data;
* tune models against the final test set;
* deploy the final application;
* select the final production model;
* perform extensive hyperparameter optimization;
* interpret synthetic correlations as medical evidence.

Those tasks belong to later sprints.

---

# 19. Sprint 1 Success Statement

Sprint 1 is successful when we can truthfully state:

> We audited and documented the 100,000-record synthetic cardiac dataset, completed reproducible exploratory analysis, established a leakage-safe Logistic Regression baseline using stratified cross-validation, recorded the ROC-AUC benchmark that future models must beat, implemented the first TensorFlow/Keras neural network without using the locked final test set for model selection, completed mentor review, and closed the sprint with documented findings and retrospective actions.
