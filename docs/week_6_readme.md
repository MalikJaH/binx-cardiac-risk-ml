# Week 6 — From Neural Network Fundamentals to Model Training

This week, I began **Phase 3 — Sprint 1** of the cardiac project and moved from the basic structure of a neural network to training and evaluating one with TensorFlow.

The week followed a clear progression:

```text
Day 1: establish the project and baseline
    ↓
Day 2: understand how a network makes a prediction
    ↓
Day 3: understand how a network learns and apply it to BRFSS 2024
```

## Day 1 — Sprint Planning and Baseline Model

I began by auditing the **100,000-record synthetic cardiac dataset**. I inspected its structure, data types, descriptive statistics, duplicates, missing values, feature ranges, categorical values, and the distribution of the `cardiac_risk` target.

Invalid values were converted to missing values so they could be handled consistently inside a preprocessing pipeline. I then saved the validated data and created an 80/20 stratified train-test split, keeping the test set separate from development decisions.

The preprocessing pipeline used:

* median imputation and `StandardScaler` for numeric features
* most-frequent imputation and `OneHotEncoder` for categorical features

I trained a Logistic Regression baseline and evaluated it with 5-fold stratified cross-validation.

| Metric | Mean CV score |
|---|---:|
| Accuracy | 0.730 |
| Precision | 0.537 |
| Recall | 0.723 |
| F1-score | 0.616 |
| ROC-AUC | **0.806 ± 0.004** |

This gave the original synthetic-data project a reproducible benchmark. I also learned how neurons use inputs, weights, biases, and activation functions inside input, hidden, and output layers.

## Day 2 — Activations, Forward Propagation and Loss

On Day 2, I learned how a network transforms inputs into a prediction.

Without non-linear activation functions, several neural-network layers would still behave like one linear model. I compared the most common activations:

* **ReLU** for efficient non-linearity in hidden layers
* **sigmoid** for one binary output
* **softmax** for multi-class output
* **tanh** for values between `-1` and `1`

I implemented ReLU, sigmoid, and tanh with NumPy and plotted how each function transforms values from `-5` to `5`.

I then calculated a complete forward pass through a small network:

```text
2 inputs
→ 2 hidden neurons
→ ReLU
→ 1 output neuron
→ sigmoid
```

For the sample `X = [0.6, -0.2]`, the output was approximately `0.5666`. This was an illustrative probability-like score for the positive class.

I also connected the architecture to binary classification:

* ReLU is appropriate for the hidden layers.
* Sigmoid is appropriate for a single binary output.
* Binary cross-entropy measures the error between the output and the true binary class.

Day 2 explained the first half of learning: **how the network makes a prediction and measures its error**.

## Day 3 — Backpropagation, Optimization and the BRFSS Project

Day 3 explained the second half: **how the network learns from its error**.

The complete training loop is:

```text
forward pass → loss calculation → backpropagation → optimizer update
```

TensorFlow automatically uses gradients and the chain rule to perform backpropagation. Adam then uses those gradients to update the weights. I also learned how epochs, batches, learning rates, and early stopping affect training.

### Moving to BRFSS 2024

I expanded the project using the real **CDC BRFSS 2024 survey extract**. The file loaded with 457,670 respondents and 301 columns.

I documented a focused design with **19 predictors and one target**, covering demographics, health status, medical history, lifestyle, socioeconomic factors, mobility, and healthcare access.

The binary target is `cardiac_disease`:

```text
0 → no reported CHD or myocardial infarction
1 → reported CHD or myocardial infarction
```

Direct source variables used to construct the target were excluded to prevent target leakage. Survey-specific missing and refusal codes were decoded per variable, BMI and alcohol measures were converted, and readable names replaced the original CDC field names.

The resulting dataset contains:

| Item | Result |
|---|---:|
| Rows | 452,464 |
| Predictors | 19 |
| Target columns | 1 |
| Class `0` | 90.64% |
| Class `1` | 9.36% |

### Training the TensorFlow Model

The data was divided into training, validation, and untouched test sets. After imputation, scaling, and one-hot encoding, each respondent had 61 model inputs.

The network used:

```text
61 inputs → 64 ReLU → 32 ReLU → 1 sigmoid
```

I compared learning rates of `1e-5`, `1e-3`, and `1e-1`. The `1e-3` run reached the lowest validation loss quickly, while later epochs began to worsen. I therefore selected Adam with a learning rate of `0.001` and used early stopping to restore the best weights.

Because only 9.36% of observations belong to class `1`, I used class weights to make minority-class errors more important during training.

The final model achieved:

| Test metric | Result |
|---|---:|
| Accuracy | 0.7172 |
| ROC-AUC | **0.8393** |
| Class-1 precision | 0.22 |
| Class-1 recall | **0.82** |
| Class-1 F1-score | 0.35 |

The class weights helped the network detect 82% of positive cases, but they also increased false positives. This result showed why an imbalanced health-related task cannot be understood from accuracy alone.

The BRFSS model is a separate experiment with a different dataset and target, so its metrics should not be treated as a direct improvement over the Day 1 synthetic baseline.

## What I Completed This Week

By the end of Week 6, I had completed:

* a validated synthetic dataset and Logistic Regression benchmark
* NumPy implementations and visualizations of common activation functions
* a manual forward pass through a small neural network
* a detailed BRFSS feature-selection and leakage-prevention document
* a reproducible BRFSS cleaning and validation notebook
* a cleaned 452,464-row cardiac-classification dataset
* a TensorFlow neural network with class weighting and early stopping
* learning-rate, loss, and ROC-AUC experiments
* final evaluation on an untouched BRFSS test set

## Main Takeaway

Week 6 connected three levels of understanding:

```text
project design and baseline comparison
            ↓
forward propagation and loss
            ↓
backpropagation and optimization
```

I no longer see a neural network as a black box. I can explain how its layers transform features, why the output and loss match the task, how gradients update the weights, and how validation curves guide training decisions.

The final BRFSS output is a **probability-like score for the reported-disease class**, not a clinically calibrated risk estimate. Because BRFSS is cross-sectional, the model classifies reported disease history rather than predicting future disease.

### Project Work

* [BRFSS feature-selection rationale](brfss_2024_cardiac_feature_selection.md)
* [BRFSS data-preparation notebook](../notebooks/neural_network/01_data_preparation.ipynb)
* [Neural-network training notebook](../notebooks/neural_network/02_neural_network.ipynb)
* [Day 3 detailed README](day_3_readme.md)
