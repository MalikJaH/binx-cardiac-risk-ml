# Day 3 — Backpropagation, Optimization & Neural Network Training

Today, I learned how a neural network moves from making predictions with random weights to improving those predictions through **backpropagation and optimization**. I also applied the full process to a new cardiac-classification dataset built from the **CDC BRFSS 2024 survey**.

## What I Learned

The main idea was the four-step training loop:

```text
forward pass → loss calculation → backpropagation → optimizer update
```

During the **forward pass**, patient features move through the network and produce an output. The loss function compares that output with the true class and measures the error.

**Backpropagation** then works backward from the loss. Using gradients and the chain rule, it calculates how much each weight contributed to the error. TensorFlow performs these calculations automatically.

Finally, the optimizer updates the weights. I used **Adam**, which adapts its updates for individual parameters and is a reliable starting point for many neural-network tasks.

I also learned the meaning of two important training terms:

* An **epoch** is one complete pass through the training dataset.
* A **batch** is the smaller group of observations processed before each weight update.

### The Learning Rate

The learning rate controls the size of the optimizer's updates. A very small value may learn slowly, while a very large value can overshoot useful solutions and make the validation loss unstable.

Instead of judging learning rates only by labels such as “low” or “high,” I learned to read their actual validation curves.

In my experiment:

* `1e-5` reduced validation loss slowly and steadily.
* `1e-3` reached the lowest validation loss quickly, but then began to worsen.
* `1e-1` showed less stable, oscillating behavior.

I selected `0.001` because it found the lowest validation loss quickly. I paired it with **early stopping**, which restores the weights from the best validation epoch instead of keeping later updates that generalize less well.

## Project Work

### Selecting a Real-World Dataset

I documented the move from the earlier synthetic project data to the **2024 Behavioral Risk Factor Surveillance System (BRFSS)** dataset.

The raw file loaded with:

* **457,670 respondents**
* **301 available columns**
* approximately **1.09 GB** of survey data

Using every variable would introduce unnecessary complexity, duplicated information, high missingness, and possible target leakage. I therefore selected **19 predictors and one binary target**.

The predictors cover:

* demographics and physical characteristics
* general, physical, and mental health
* diabetes, stroke, and kidney-disease history
* physical activity, smoking, and alcohol use
* education, income, and employment
* mobility and access to healthcare

The target is derived from `_MICHD` and renamed to `cardiac_disease`:

```text
0 → no reported history of coronary heart disease or myocardial infarction
1 → reported history of coronary heart disease or myocardial infarction
```

I intentionally excluded `CVDINFR4` and `CVDCRHD4` because they are used to construct `_MICHD`. Including them would give the model information that directly defines the answer, causing **target leakage** and unrealistically strong results.

### Cleaning and Preparing the BRFSS Data

BRFSS survey codes require variable-specific cleaning. Values such as `7`, `9`, `77`, `88`, `99`, `777`, and `999` can mean “don't know,” “refused,” “none,” or missing depending on the question. I therefore decoded each selected variable separately rather than replacing the same number globally.

The preparation notebook:

* removed 5,206 observations without a valid target
* converted BMI from the stored hundredths format to standard BMI units
* converted poor-health-day codes into values from 0 to 30
* converted weekly and monthly alcohol codes into estimated days per month
* decoded invalid or refused categorical responses as missing values
* renamed the CDC fields to readable project names
* checked missingness, numeric ranges, categorical values, duplicates, and target balance

The cleaned dataset contains:

| Item | Result |
|---|---:|
| Rows | 452,464 |
| Columns | 20 |
| Class `0` | 410,126 (90.64%) |
| Class `1` | 42,338 (9.36%) |
| Exact duplicate rows found | 1,063 |

The duplicate count was documented rather than silently removed because repeated survey-response combinations do not necessarily prove that the records are erroneous duplicates.

### Building the Neural Network

I kept the final test set untouched and created separate training and validation sets. Numeric features use median imputation and standardization, while categorical features use most-frequent imputation and one-hot encoding.

After preprocessing, each respondent is represented by **61 numeric inputs**:

| Split | Rows |
|---|---:|
| Training | 316,724 |
| Validation | 45,247 |
| Test | 90,493 |

The TensorFlow model contains:

```text
61 inputs
→ 64 ReLU neurons
→ 32 ReLU neurons
→ 1 sigmoid output
```

The network has **6,081 trainable parameters**. ReLU gives the hidden layers the non-linearity needed to learn feature interactions. The single sigmoid neuron produces a probability-like score for `cardiac_disease = 1`, and `binary_crossentropy` measures error because this is a binary-classification task.

Before training, one validation respondent received a score of `0.5494` from randomly initialized weights. After training, the score became `0.0478`; the respondent's true class was `0`. This example shows that training changed the forward pass, although overall performance must be judged across the full dataset.

### Handling Class Imbalance

Only 9.36% of respondents belong to class `1`, so accuracy alone could hide poor minority-class detection. I used balanced class weights:

```text
class 0 → 0.552
class 1 → 5.344
```

This makes errors on the less common positive class count more heavily. The expected trade-off is higher class-1 recall with more false-positive predictions.

### Final Results

Early stopping ended training after four epochs and restored the best validation-loss weights.

The untouched test set produced:

| Metric | Result |
|---|---:|
| Test loss | 0.5100 |
| Test accuracy | 0.7172 |
| Test ROC-AUC | 0.8393 |
| Class-1 precision | 0.22 |
| Class-1 recall | 0.82 |
| Class-1 F1-score | 0.35 |

The model found **82% of the reported cardiac-disease cases** at the `0.5` threshold, but only 22% of its positive predictions were correct. This is consistent with weighting the minority class more heavily: the model misses fewer positive cases but creates more false alarms.

The sigmoid output is a **probability-like score for class 1**, not automatically a calibrated clinical risk estimate. BRFSS is also cross-sectional, so this project classifies reported disease history; it does not predict who will develop heart disease in the future.

## Uncommitted Project Files Covered

| File | Work completed |
|---|---|
| [`data/LLCP2024/raw/LLCP2024.XPT`](../data/LLCP2024/raw/LLCP2024.XPT) | Original CDC BRFSS 2024 survey extract |
| [`data/LLCP2024/processed/brfss_2024_cardiac_clean.csv`](../data/LLCP2024/processed/brfss_2024_cardiac_clean.csv) | Cleaned 452,464-row, 20-column cardiac dataset |
| [`docs/brfss_2024_cardiac_feature_selection.md`](brfss_2024_cardiac_feature_selection.md) | Feature rationale, target definition, leakage safeguards, cleaning rules, and modeling recommendations |
| [`notebooks/neural_network/01_data_preparation.ipynb`](../notebooks/neural_network/01_data_preparation.ipynb) | BRFSS selection, decoding, validation, and export workflow |
| [`notebooks/neural_network/02_neural_network.ipynb`](../notebooks/neural_network/02_neural_network.ipynb) | Preprocessing, forward pass, training, learning-rate experiment, early stopping, and final evaluation |

Overall, Day 3 connected the mathematics of learning to the complete project workflow. I learned how backpropagation calculates gradients, how Adam updates weights, and how the learning rate changes training behavior. In the repository, I turned the raw BRFSS survey into a leakage-aware cardiac dataset and trained a neural network whose limitations and class trade-offs are clearly documented.
