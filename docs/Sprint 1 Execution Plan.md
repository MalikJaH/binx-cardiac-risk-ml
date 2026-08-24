# Sprint 1 Execution Plan

**Project:** Cardiac Patient Monitoring — Synthetic Cardiac-Risk Classification
**Phase:** Phase 3 Capstone
**Sprint:** Sprint 1 of 4
**Week:** Week 6
**Duration:** 5 working days — 40 hours
**Stack:** Python, NumPy, Pandas, scikit-learn, Matplotlib, TensorFlow/Keras, Jupyter, GitHub

---

# 1. Sprint Goal

Establish a reproducible classical machine-learning baseline, understand the mechanics of neural-network training, and build the first TensorFlow/Keras neural network for the cardiac-risk classification project.

By the end of Sprint 1, the project should contain:

* a validated project dataset;
* brief documented EDA;
* a leakage-safe Logistic Regression baseline;
* a recorded ROC-AUC benchmark;
* activation-function and forward-pass experiments;
* training-mechanics experiments;
* a first TensorFlow/Keras neural network;
* training and validation curves;
* controlled tuning experiments;
* comparison against the baseline;
* mentor review evidence;
* Sprint Review and Retrospective.

The purpose of Sprint 1 is to establish the complete workflow.

The neural network should be compared fairly with the baseline. Beating the baseline is desirable, but a lower score is still a valid result if the methodology and analysis are correct.

---

# 2. Dataset

Sprint 1 uses:

```text
synthetic_cardiac_100k.csv
```

Target:

```text
cardiac_risk
```

Target meaning:

```text
0 = lower simulated cardiac-risk pattern
1 = elevated simulated cardiac-risk pattern
```

Important dataset properties:

* approximately 100,000 rows;
* synthetic data only;
* binary classification target;
* intentional missing values;
* exact duplicate rows;
* intentionally invalid/outlier values;
* no real clinical or patient data.

The project must clearly state that patterns discovered in this dataset are properties of the synthetic data-generation process and must not be interpreted as clinical evidence.

---

# 3. Evaluation Strategy

## Primary metric

```text
ROC-AUC
```

The official baseline benchmark is:

```text
Mean 5-Fold Stratified CV ROC-AUC ± Standard Deviation
```

Supporting metrics:

* Accuracy
* Precision
* Recall
* F1-score

The baseline benchmark should come from actual notebook execution and must never be written in advance.

Example:

```text
Logistic Regression
Mean CV ROC-AUC = X.XXX ± X.XXX
```

Future models will be compared primarily against this benchmark.

---

# 4. Data-Splitting Rule

Create a stratified development/test split.

The final test set must not influence:

* architecture selection;
* learning-rate selection;
* dropout selection;
* batch-size selection;
* epoch selection;
* hyperparameter tuning.

Use training/validation data for model development.

Prefer one final test evaluation after model decisions are complete.

If the internship exercise specifically requires an earlier test evaluation, record it once and do not use the result to select or tune later configurations.

---

# 5. Repository Structure

```text
cardiac-risk-capstone/
│
├── data/
│   ├── new/
│   │   └── synthetic_cardiac_100k.csv
│   ├── images/
│   └── data_dictionary.md
│
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_logistic_regression_baseline.ipynb
│   ├── 03_activations_forward_pass.ipynb
│   ├── 04_training_mechanics.ipynb
│   └── 05_keras_neural_network.ipynb
│
├── reports/
│   ├── baseline_cv_metrics.csv
│   ├── sprint1_findings.md
│   └── sprint1_retrospective.md
│
├── models/
│
├── src/
│   └── validation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

Not every folder must contain substantial code during Sprint 1. The priority is a clear and reproducible notebook workflow.

---

# 6. Sprint Backlog

| ID    | Task                                        | Priority |
| ----- | ------------------------------------------- | -------- |
| S1-01 | Sprint planning and repository setup        | Must     |
| S1-02 | Dataset validation and brief EDA            | Must     |
| S1-03 | Logistic Regression baseline                | Must     |
| S1-04 | Record official baseline benchmark          | Must     |
| S1-05 | Activation-function experiments             | Must     |
| S1-06 | Manual/NumPy forward propagation            | Must     |
| S1-07 | Training-loop and learning-rate experiments | Must     |
| S1-08 | Mentor review                               | Must     |
| S1-09 | First TensorFlow/Keras network              | Must     |
| S1-10 | Dropout / Batch Normalization experiment    | Must     |
| S1-11 | Controlled tuning and EarlyStopping         | Must     |
| S1-12 | Baseline vs neural-network comparison       | Must     |
| S1-13 | Sprint Review and Retrospective             | Must     |

---

# 7. Day 1 — Sprint Planning, Dataset, Baseline & Neural-Network Architecture

**Goal:** Establish the project foundation and create the benchmark that every later model will be compared against.

## 7.1 Sprint Planning

Confirm:

* dataset;
* target;
* problem type;
* primary metric;
* Sprint goal;
* modelling restrictions;
* Definition of Done.

Use the existing Sprint feature branch:

```bash
feature/sprint-1-baseline
```

If the branch already exists, continue using it rather than creating another branch unnecessarily.

---

## 7.2 Dataset Validation

Audit:

* shape;
* data types;
* missing values;
* duplicates;
* invalid numeric values;
* invalid categories;
* target balance;
* identifiers.

Remove exact duplicates before modelling.

Convert invalid measurements to missing values according to documented validation rules.

Learned imputation must occur inside the preprocessing pipeline.

---

## 7.3 Brief EDA

The purpose of Day 1 EDA is to understand the dataset sufficiently for modelling, not to spend the whole sprint producing visualizations.

Include:

* target distribution;
* important numeric distributions;
* categorical distributions;
* missing-value summary;
* several useful feature/target relationships;
* numeric correlation overview.

Document at least three meaningful observations.

State the synthetic-data limitation.

---

## 7.4 Baseline Model

Build:

```text
Numeric features
→ MedianImputer
→ StandardScaler

Categorical features
→ MostFrequentImputer
→ OneHotEncoder

→ ColumnTransformer
→ LogisticRegression
```

Use:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Record:

* ROC-AUC mean;
* ROC-AUC standard deviation;
* accuracy;
* precision;
* recall;
* F1.

Write the official baseline benchmark in Markdown.

---

## 7.5 Neural-Network Architecture Foundations

Understand:

```text
Neuron
Weights
Bias
Input layer
Hidden layer
Output layer
```

Core operation:

```text
z = XW + b
output = activation(z)
```

No TensorFlow model is required on Day 1.

---

## Day 1 Definition of Done

* [x] Sprint plan confirmed.
* [x] Dataset loaded.
* [x] Dataset audit documented.
* [x] Important validation issues handled.
* [x] Brief EDA completed.
* [x] Train/test strategy established.
* [x] Logistic Regression pipeline implemented.
* [x] Stratified 5-fold CV executed.
* [x] Baseline metrics recorded.
* [x] Baseline ROC-AUC benchmark documented.
* [x] Work pushed to the Sprint branch.
* [x] Draft PR opened.

---

# 8. Day 2 — Activations, Forward Propagation & Loss

**Goal:** Understand how a neural network transforms an input into a prediction and how the prediction error is measured.

## 8.1 Activation Functions

Study:

```text
ReLU
Sigmoid
Tanh
Softmax
```

Understand why non-linear activation functions are necessary.

Practical rule:

```text
Hidden layers → ReLU

Binary classification → Sigmoid
Multi-class classification → Softmax
Regression → Linear output
```

---

## 8.2 Activation Experiment

Create:

```text
notebooks/03_activations_forward_pass.ipynb
```

Plot:

* ReLU;
* sigmoid;
* tanh;

over a range such as:

```text
-5 to 5
```

Add a short Markdown interpretation explaining how each function transforms its input.

---

## 8.3 Project Output Activation and Loss

The cardiac project is binary classification.

Therefore:

```text
Output activation = Sigmoid
Loss = Binary Cross-Entropy
```

Document why:

* sigmoid returns a value between 0 and 1;
* the result can represent the predicted probability of `cardiac_risk = 1`;
* binary cross-entropy is appropriate for a binary target.

---

## 8.4 Forward Propagation Experiment

Create a tiny NumPy network:

```text
2 inputs
→ 2 hidden neurons
→ ReLU
→ 1 output neuron
→ Sigmoid
```

Calculate:

```text
z1 = XW1 + b1
a1 = ReLU(z1)

z2 = a1W2 + b2
prediction = sigmoid(z2)
```

Record the intermediate values and final prediction.

---

## Day 2 Definition of Done

* [x] ReLU plotted.
* [x] Sigmoid plotted.
* [x] Tanh plotted.
* [x] Activation observations documented.
* [x] Sigmoid justified for the project output.
* [x] Binary cross-entropy justified.
* [x] Tiny NumPy forward pass completed.
* [x] Intermediate values documented.
* [x] Final prediction interpreted.
* [x] Notebook restarted and Run All succeeds.
* [x] Changes committed.
* [x] Changes pushed to Draft PR.


# 9. Day 3 — Backpropagation, Gradient Descent & Optimizers

**Goal:** Understand how the network moves from making a prediction to learning from its error.

Create:

```text
notebooks/04_training_mechanics.ipynb
```

---

## 9.1 Training Loop

Document:

```text
Forward pass
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
Optimizer update
    ↓
New weights
```

Explain in your own words:

* what backpropagation computes;
* why gradients are needed;
* how gradient descent updates weights;
* why the chain rule is involved.

No manual implementation of full backpropagation is required.

---

## 9.2 Learning Rate

Understand:

```text
Too low
→ very slow learning

Too high
→ unstable/diverging loss

Reasonable
→ steady reduction in loss
```

Run a small experiment using three learning rates representing:

* too low;
* reasonable;
* too high.

Plot the resulting loss curves.

The purpose is to observe behaviour, not to maximize performance.

---

## 9.3 Optimizers

Understand:

### SGD

Basic gradient-based parameter updates.

### Adam

Adaptive learning rates per parameter and a strong general-purpose default.

Also understand:

* epoch;
* batch;
* batch size.

---

# Mentor Review — Day 3

The Draft PR should now contain:

* dataset audit;
* EDA;
* preprocessing;
* Logistic Regression baseline;
* baseline metrics;
* activation-function notebook;
* forward-pass experiment;
* training-mechanics notebook.

Update the PR description with:

```markdown
## Sprint Goal

## Completed

## Dataset Findings

## Baseline Result

## Deep-Learning Concepts Completed

## Current Experiments

## Questions / Blockers
```

After mentor feedback:

* read every comment;
* address methodological issues;
* rerun affected notebook sections;
* reply to important review comments;
* push fixes.

---

## Day 3 Definition of Done

* [ ] Four-step training loop documented.
* [ ] Backpropagation explained conceptually.
* [ ] Gradient descent understood.
* [ ] Learning rate explained.
* [ ] Three learning-rate behaviours tested.
* [ ] Loss curves plotted.
* [ ] Adam vs SGD understood.
* [ ] Epoch and batch terminology understood.
* [ ] Draft PR ready for mentor review.
* [ ] Mentor feedback addressed where available.

---

# 10. Day 4 — Build & Train the First TensorFlow/Keras Network

**Goal:** Apply Days 1–3 by building and training an actual neural network.

Create:

```text
notebooks/05_keras_neural_network.ipynb
```

Use the same project data and compatible preprocessing strategy as the baseline.

---

## 10.1 First Architecture

Start simple:

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
```

The exact hidden-layer width is an initial experiment, not a final architecture.

---

## 10.2 Compile

Use:

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

This connects directly to the Day 2 decisions:

```text
Binary task
→ Sigmoid output
→ Binary Cross-Entropy
```

---

## 10.3 Train

Train using training data with a separate validation set.

Start with at least:

```text
30 epochs
batch_size = 32
```

Record:

* training loss;
* validation loss;
* training ROC-AUC;
* validation ROC-AUC;
* accuracy.

Plot:

```text
Training vs Validation Loss
Training vs Validation ROC-AUC
```

Diagnose:

* underfitting;
* reasonable fit;
* overfitting.

---

## 10.4 Dropout / Batch Normalization

Run one controlled architecture improvement.

Example:

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
```

Compare the curves against the original model.

Do not make many changes simultaneously.

---

## Day 4 Definition of Done

* [ ] TensorFlow/Keras model builds.
* [ ] Correct sigmoid output used.
* [ ] Binary cross-entropy used.
* [ ] Adam used.
* [ ] Network trains successfully.
* [ ] Training loss plotted.
* [ ] Validation loss plotted.
* [ ] ROC-AUC curves plotted.
* [ ] Fit diagnosed in Markdown.
* [ ] Dropout and/or BatchNormalization tested.
* [ ] Controlled comparison documented.
* [ ] No test-set-driven tuning performed.

Suggested commit:

```bash
git add .
git commit -m "feat: train first Keras neural network"
git push
```

---

# 11. Day 5 — Tuning, Evaluation & Sprint Review

**Goal:** Improve the network systematically, evaluate the final Sprint 1 candidate, and close the sprint with reproducible evidence.

---

## 11.1 Controlled Tuning

Tune one variable at a time.

Priority:

```text
1. Learning rate
2. Network width/depth
3. Dropout
4. Batch size
```

Maintain an experiment table:

| Run          | Architecture | Learning Rate | Dropout | Batch Size | Validation ROC-AUC | Notes   |
| ------------ | ------------ | ------------: | ------: | ---------: | -----------------: | ------- |
| Baseline NN  | 64 → 32 → 1  |         0.001 |       0 |         32 |                ... | Initial |
| Experiment 1 | ...          |           ... |     ... |        ... |                ... | ...     |
| Experiment 2 | ...          |           ... |     ... |        ... |                ... | ...     |

Do not run dozens of arbitrary configurations.

---

## 11.2 EarlyStopping

Use:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Confirm:

* training stops when validation loss stops improving;
* best weights are restored.

---

## 11.3 Final Sprint Comparison

Compare:

| Model               | Evaluation                     |       ROC-AUC |
| ------------------- | ------------------------------ | ------------: |
| Logistic Regression | Mean 5-fold CV                 | X.XXX ± X.XXX |
| Neural Network      | Validation / agreed evaluation |         X.XXX |

Do not pretend these numbers are directly equivalent if the validation procedures differ.

Clearly document the methodology used for each score.

---

## 11.4 Final Test Evaluation

Once model decisions are complete, perform one final test evaluation if required for Sprint 1.

Record:

* ROC-AUC;
* accuracy;
* precision;
* recall;
* F1;
* confusion matrix if useful.

Do not return to tuning based on the final test result.

---

# 12. Reproducibility Check

For every notebook:

```text
Kernel
→ Restart
→ Run All
```

Required notebooks:

```text
01_data_audit_eda.ipynb
02_logistic_regression_baseline.ipynb
03_activations_forward_pass.ipynb
04_training_mechanics.ipynb
05_keras_neural_network.ipynb
```

All notebooks should execute without relying on hidden notebook state.

---

# 13. Sprint Review

Demo:

1. project dataset;
2. dataset-quality issues;
3. important EDA findings;
4. preprocessing pipeline;
5. Logistic Regression baseline;
6. baseline ROC-AUC;
7. activation-function experiment;
8. forward propagation;
9. training loop;
10. first Keras neural network;
11. training/validation curves;
12. tuning experiments;
13. baseline vs neural-network comparison;
14. GitHub PR.

Be prepared to answer:

```text
What is the target?

Why is sigmoid used?

Why is binary cross-entropy used?

What happens during forward propagation?

What does backpropagation calculate?

What does the learning rate control?

Why was Adam selected?

What is the baseline?

What ROC-AUC must later models beat?

Did the neural network outperform the baseline?

Was the final test set protected from model selection?

What should improve in Sprint 2?
```

---

# 14. Sprint Retrospective

Create:

```text
reports/sprint1_retrospective.md
```

Document:

## What went well?

Actual successful parts of the sprint.

## What did not go well?

Real problems encountered.

## What should improve?

One or two concrete improvements.

## Sprint 2 Action

Choose one specific behavioural or engineering improvement.

Example:

> Record every experiment's configuration and validation result immediately so model comparisons remain reproducible.

---

# 15. Git Workflow

Continue using:

```text
feature/sprint-1-baseline
```

Recommended history:

```text
feat: add dataset audit and EDA
feat: add leakage-safe logistic regression baseline
feat: add activation and forward propagation lab
feat: add neural network training mechanics
fix: address mentor review feedback
feat: train first Keras neural network
feat: add neural network tuning experiments
docs: complete Sprint 1 review and retrospective
```

Push regularly.

Do not wait until Day 5 to commit all work.

---

# 16. Sprint 1 Definition of Done

## Dataset & Baseline

* [ ] Dataset documented.
* [ ] Data-quality issues audited.
* [ ] Brief EDA completed.
* [ ] Synthetic-data limitation documented.
* [ ] Leakage-safe preprocessing used.
* [ ] Logistic Regression baseline completed.
* [ ] Stratified 5-fold CV completed.
* [ ] ROC-AUC benchmark recorded.

## Deep-Learning Foundations

* [ ] Activation functions understood.
* [ ] Activation functions plotted.
* [ ] Sigmoid output justified.
* [ ] Binary cross-entropy justified.
* [ ] Forward pass implemented in NumPy.
* [ ] Training loop documented.
* [ ] Backpropagation understood conceptually.
* [ ] Learning-rate behaviour demonstrated.

## TensorFlow/Keras

* [ ] Sequential network built.
* [ ] ReLU hidden layers used.
* [ ] Sigmoid output used.
* [ ] Binary cross-entropy used.
* [ ] Adam optimizer used.
* [ ] Training and validation curves plotted.
* [ ] Fit diagnosed.
* [ ] Dropout and/or BatchNormalization tested.
* [ ] EarlyStopping implemented.
* [ ] Controlled tuning completed.

## Evaluation

* [ ] Neural network compared against baseline.
* [ ] Validation methodology documented.
* [ ] Final test set not used for model selection.
* [ ] Final evaluation recorded when appropriate.

## Engineering

* [ ] Work committed regularly.
* [ ] Draft PR maintained.
* [ ] Mentor review completed.
* [ ] Review comments addressed.
* [ ] Every notebook runs top-to-bottom.
* [ ] README updated.
* [ ] Sprint Review completed.
* [ ] Sprint Retrospective completed.
* [ ] PR approved before merge.

---

# 17. Current Status — End of Day 2

Based on work completed so far:

```text
DAY 1
✅ Sprint planning
✅ Dataset work
✅ Baseline established
✅ Draft PR opened

DAY 2
✅ ReLU / sigmoid / tanh implemented
✅ Activation functions plotted
✅ Sigmoid selected for cardiac-risk output
✅ Binary cross-entropy selected and justified
✅ Tiny NumPy forward pass implemented
✅ Forward-pass result documented
⬜ Restart notebook and Run All
⬜ Commit
⬜ Push

DAY 3
Next → Backpropagation, gradient descent, optimizers, learning-rate experiment, mentor review

DAY 4
Next → TensorFlow/Keras network

DAY 5
Next → Tuning, EarlyStopping, evaluation, Sprint Review and Retrospective
```

---

# Sprint 1 Success Statement

Sprint 1 is successful when we can state:

> We validated and documented the synthetic cardiac-risk dataset, established a reproducible Logistic Regression ROC-AUC baseline, demonstrated activation functions and forward propagation, understood the mechanics of backpropagation and optimization, built and trained our first TensorFlow/Keras neural network, evaluated its training behaviour, compared it fairly with the baseline, completed mentor review, and closed the sprint with reproducible notebooks and documented results.
