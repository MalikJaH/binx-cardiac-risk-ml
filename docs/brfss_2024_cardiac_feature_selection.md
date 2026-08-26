# BRFSS 2024 — Final 20-Column Cardiac Classification Dataset

## 1. Project Goal

The original **CDC BRFSS 2024** dataset contains approximately:

* **457,670 respondents**
* **345 variables**
* Real survey data collected by the CDC

For this machine-learning project, using all 345 variables would add unnecessary complexity, duplicated information, high missingness, and potential target leakage.

Instead, we will construct a focused dataset containing exactly:

* **19 predictor features**
* **1 target variable**
* **20 columns total**

The task is:

> **Binary classification of whether a respondent reports a history of coronary heart disease or myocardial infarction using demographic, lifestyle, health-status, and comorbidity information.**

The selected variables are all present in the official BRFSS 2024 dataset.

---

# 2. Final 20 Columns

```python
selected_columns = [
    "_AGE80",
    "_SEX",
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "CHECKUP1",
    "_TOTINDA",
    "DIABETE4",
    "CVDSTRK3",
    "CHCKDNY2",
    "_SMOKER3",
    "ALCDAY4",
    "_EDUCAG",
    "INCOME3",
    "EMPLOY1",
    "DIFFWALK",
    "PERSDOC3",
    "MEDCOST1",
    "_MICHD"
]
```

These give:

```text
19 input features
+
1 target
=
20 total columns
```

---

# 3. Summary of the Selected Variables

| #  | BRFSS Variable | Friendly Name               | Category          | Role       |
| -- | -------------- | --------------------------- | ----------------- | ---------- |
| 1  | `_AGE80`       | Age                         | Demographic       | Feature    |
| 2  | `_SEX`         | Sex                         | Demographic       | Feature    |
| 3  | `_BMI5`        | BMI                         | Physical          | Feature    |
| 4  | `GENHLTH`      | General Health              | Health Status     | Feature    |
| 5  | `PHYSHLTH`     | Poor Physical Health Days   | Health Status     | Feature    |
| 6  | `MENTHLTH`     | Poor Mental Health Days     | Health Status     | Feature    |
| 7  | `CHECKUP1`     | Last Routine Checkup        | Healthcare        | Feature    |
| 8  | `_TOTINDA`     | Physical Activity           | Lifestyle         | Feature    |
| 9  | `DIABETE4`     | Diabetes Status             | Medical           | Feature    |
| 10 | `CVDSTRK3`     | Stroke History              | Medical           | Feature    |
| 11 | `CHCKDNY2`     | Kidney Disease              | Medical           | Feature    |
| 12 | `_SMOKER3`     | Smoking Status              | Lifestyle         | Feature    |
| 13 | `ALCDAY4`      | Alcohol Consumption         | Lifestyle         | Feature    |
| 14 | `_EDUCAG`      | Education Level             | Socioeconomic     | Feature    |
| 15 | `INCOME3`      | Income Level                | Socioeconomic     | Feature    |
| 16 | `EMPLOY1`      | Employment Status           | Socioeconomic     | Feature    |
| 17 | `DIFFWALK`     | Difficulty Walking          | Functional Health | Feature    |
| 18 | `PERSDOC3`     | Personal Doctor             | Healthcare        | Feature    |
| 19 | `MEDCOST1`     | Cost Prevented Medical Care | Healthcare        | Feature    |
| 20 | `_MICHD`       | CHD / Myocardial Infarction | Target            | **Target** |

---

# 4. Target Variable

## `_MICHD` — Coronary Heart Disease / Myocardial Infarction

### What it represents

`_MICHD` is the target of our binary classification problem.

It represents whether the respondent reports having had:

* myocardial infarction / heart attack, or
* coronary heart disease.

The variable is calculated by the CDC using responses to cardiovascular questions. `_MICHD` is included as a calculated variable in the 2024 dataset.

### Why I selected it

Our project needs a clear binary cardiac outcome.

This gives us:

```text
No reported CHD/MI
vs.
Reported CHD/MI
```

which is appropriate for:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Neural Networks
* Binary Cross-Entropy
* ROC-AUC
* Precision
* Recall
* F1-score

### Machine-learning encoding

We should convert:

```text
CDC value 1 → 1 → CHD/MI reported
CDC value 2 → 0 → CHD/MI not reported
```

For example:

```python
cardiac = cardiac[
    cardiac["_MICHD"].isin([1, 2])
].copy()

cardiac["cardiac_disease"] = cardiac["_MICHD"].map({
    1: 1,
    2: 0
})
```

Then drop `_MICHD`.

```python
cardiac.drop(columns="_MICHD", inplace=True)
```

---

# 5. CRITICAL — Variables We Must NOT Use

The raw BRFSS dataset contains:

```text
CVDINFR4
CVDCRHD4
```

These appear in the 2024 dataset immediately before other cardiovascular disease variables.

We intentionally exclude them.

## Why?

Because `_MICHD` is calculated using these variables.

Conceptually:

```text
CVDINFR4
      \
       → _MICHD
      /
CVDCRHD4
```

If we used:

```python
X = [
    "CVDINFR4",
    "CVDCRHD4"
]
```

to predict:

```python
y = df["_MICHD"]
```

the model would effectively receive the answer.

This is called:

# Target Leakage

It could produce artificially excellent metrics such as:

```text
Accuracy = 99%
ROC-AUC = 0.99
```

without representing a meaningful predictive model.

Therefore:

```text
CVDINFR4 ❌ EXCLUDE

CVDCRHD4 ❌ EXCLUDE
```

---

# 6. Predictor 1 — `_AGE80`

## Friendly name

**Age**

## Category

Demographic

## What it represents

`_AGE80` is a calculated age variable available in BRFSS 2024. It provides respondent age while top-coding older ages according to the BRFSS calculated-variable definition. The official variable layout includes `_AGE80`.

## Why I selected it

Age is one of the strongest variables associated with cardiovascular disease.

Cardiovascular risk generally rises considerably with age because of cumulative changes involving:

* arteries
* blood vessels
* metabolic health
* chronic disease exposure
* blood pressure
* lifestyle exposure

Without age, the model would be missing one of its most fundamental explanatory variables.

## ML type

```text
Numerical
```

## Suggested processed name

```text
age
```

## Processing

Usually keep as numeric.

```python
cardiac["age"] = cardiac["_AGE80"]
```

---

# 7. Predictor 2 — `_SEX`

## Friendly name

**Sex**

## Category

Demographic

## What it represents

A calculated sex variable included in BRFSS 2024. `_SEX` appears among the demographic calculated variables in the official data layout.

## Why I selected it

Cardiovascular disease prevalence and presentation differ across sexes.

Sex can therefore contribute predictive information alongside:

```text
age
BMI
diabetes
smoking
physical activity
```

It is an important demographic control variable.

## ML type

```text
Categorical
```

## Suggested processed name

```text
sex
```

Do not treat category numbers as continuous quantities.

---

# 8. Predictor 3 — `_BMI5`

## Friendly name

**Body Mass Index**

## Category

Physical / anthropometric

## What it represents

`_BMI5` is the CDC-calculated BMI variable.

It is calculated from weight and height. CDC documentation describes it as BMI in kilograms per meter squared with two implied decimal places.

For example:

```text
2750
```

means approximately:

```text
27.50 kg/m²
```

## Why I selected it

BMI captures information related to body composition and obesity.

Higher BMI may be associated with cardiovascular risk through relationships with:

* hypertension
* diabetes
* metabolic disease
* inflammation
* dyslipidemia

It gives the model important physical-health information.

## Why use `_BMI5` instead of height + weight?

Because `_BMI5` already combines them meaningfully.

Using:

```text
weight
height
BMI
BMI category
```

all together can add redundant information.

Therefore we keep:

```text
_BMI5
```

and exclude the redundant versions.

## ML type

```text
Numerical continuous
```

## Suggested name

```text
bmi
```

## Processing

Confirm the representation after reading the XPT file. If it arrives with implied decimals, convert accordingly.

---

# 9. Predictor 4 — `GENHLTH`

## Friendly name

**General Health**

## Category

Self-reported health

## What it represents

The respondent's overall assessment of their health.

It is one of the core health-status variables in BRFSS 2024.

Typical categories correspond to concepts such as:

```text
Excellent
Very good
Good
Fair
Poor
```

## Why I selected it

General health acts as a broad summary of the respondent's overall condition.

People experiencing substantial chronic disease often report worse general health.

It can capture information that individual medical variables may miss.

## Important interpretation warning

It is:

```text
self-reported
```

not a clinical measurement.

Therefore it is useful for prediction but should not be interpreted as an objective clinical test.

## ML type

```text
Ordinal categorical
```

## Suggested name

```text
general_health
```

---

# 10. Predictor 5 — `PHYSHLTH`

## Friendly name

**Poor Physical Health Days**

## Category

Physical health

## What it represents

Number of recent days during which the respondent reported that their physical health was not good.

`PHYSHLTH` is part of the core BRFSS 2024 dataset.

## Why I selected it

This gives substantially more detail than simply asking whether someone has a diagnosed condition.

It can capture:

* physical limitations
* chronic illness burden
* poor overall health
* functional health problems

which may be associated with cardiovascular disease.

## ML type

```text
Numerical / count
```

## Suggested name

```text
poor_physical_health_days
```

## Important cleaning issue

Special values such as:

```text
77
88
99
```

may represent concepts such as:

```text
Don't know
None
Refused
```

depending on the variable definition.

Do not automatically interpret `88` as eighty-eight unhealthy days.

---

# 11. Predictor 6 — `MENTHLTH`

## Friendly name

**Poor Mental Health Days**

## Category

Mental / overall health

## What it represents

Number of recent days during which mental health was reported as not good.

`MENTHLTH` is directly present in the 2024 dataset.

## Why I selected it

Mental health is not a direct cardiac measurement, but it adds important information regarding:

* stress
* health behavior
* physical activity
* smoking
* healthcare usage
* general health status

It can contribute information not represented by BMI or diabetes alone.

## ML type

```text
Numerical / count
```

## Suggested name

```text
poor_mental_health_days
```

---

# 12. Predictor 7 — `CHECKUP1`

## Friendly name

**Time Since Last Routine Checkup**

## Category

Healthcare utilization

## What it represents

Information about how recently a respondent had a routine medical checkup.

`CHECKUP1` is part of the core BRFSS questionnaire.

## Why I selected it

Healthcare utilization can affect:

* disease detection
* treatment
* preventive care
* chronic disease management

Someone who regularly visits healthcare providers may have a very different health profile from someone who has not received routine care for many years.

## ML type

```text
Ordinal categorical
```

## Suggested name

```text
last_checkup
```

---

# 13. Predictor 8 — `_TOTINDA`

## Friendly name

**Physical Activity**

## Category

Lifestyle

## What it represents

`_TOTINDA` is a calculated BRFSS variable related to whether the respondent participated in physical activity or exercise.

It is included among the calculated variables in the 2024 data file.

## Why I selected it

Physical inactivity is an important lifestyle factor related to:

* obesity
* cardiovascular fitness
* diabetes
* metabolic health
* cardiovascular disease

It gives us behavioral information rather than only demographic or disease information.

## ML type

```text
Categorical / binary
```

## Suggested name

```text
physical_activity
```

---

# 14. Predictor 9 — `DIABETE4`

## Friendly name

**Diabetes Status**

## Category

Medical history

## What it represents

Whether the respondent reports diabetes, with additional BRFSS categories potentially distinguishing related states.

`DIABETE4` is explicitly present in the BRFSS 2024 dataset.

## Why I selected it

Diabetes is one of the strongest clinically meaningful features in this dataset for cardiovascular classification.

It is associated with:

* vascular damage
* atherosclerosis
* metabolic abnormalities
* cardiovascular complications

A cardiac model without diabetes information would discard an important known risk factor.

## ML type

```text
Categorical
```

## Suggested name

```text
diabetes_status
```

---

# 15. Predictor 10 — `CVDSTRK3`

## Friendly name

**History of Stroke**

## Category

Cardiovascular comorbidity

## What it represents

Whether the respondent reports having previously experienced a stroke.

It is included in the 2024 dataset next to the cardiovascular disease variables.

## Why I selected it

Stroke and coronary cardiovascular disease share many underlying risk factors, including:

* vascular disease
* smoking
* diabetes
* age
* metabolic health

A history of stroke may therefore provide useful information about overall vascular disease burden.

## Is this leakage?

No.

It is **not used to calculate `_MICHD`**.

However, it is a cardiovascular comorbidity, so interpretation should remain careful.

## ML type

```text
Categorical / binary
```

## Suggested name

```text
stroke_history
```

---

# 16. Predictor 11 — `CHCKDNY2`

## Friendly name

**Chronic Kidney Disease**

## Category

Medical history

## What it represents

Whether the respondent reports kidney disease.

The variable appears in the 2024 BRFSS core health section.

## Why I selected it

Kidney and cardiovascular health are strongly interconnected.

Kidney disease commonly appears alongside conditions involving:

* hypertension
* diabetes
* vascular disease
* metabolic dysfunction

Therefore it can add meaningful predictive information.

## ML type

```text
Categorical / binary
```

## Suggested name

```text
kidney_disease
```

---

# 17. Predictor 12 — `_SMOKER3`

## Friendly name

**Smoking Status**

## Category

Lifestyle

## What it represents

The CDC calculated smoking-status variable.

It is derived from smoking questions and categorizes respondents into groups such as:

```text
Everyday smoker
Someday smoker
Former smoker
Never smoker
```

The 2024 dataset includes `_SMOKER3`.

## Why I selected it

Smoking is one of the clearest modifiable cardiovascular risk factors.

It affects:

* blood vessels
* atherosclerosis
* oxygen delivery
* blood clot formation
* cardiovascular stress

It is therefore extremely appropriate for this project.

## Why use `_SMOKER3` instead of multiple smoking questions?

The raw dataset also contains:

```text
SMOKE100
SMOKDAY2
```

which are used to build smoking-related variables.

Using `_SMOKER3` gives us a clean consolidated smoking variable.

## ML type

```text
Categorical
```

## Suggested name

```text
smoking_status
```

---

# 18. Predictor 13 — `ALCDAY4`

## Friendly name

**Alcohol Consumption Frequency**

## Category

Lifestyle

## What it represents

Information about how frequently the respondent consumes alcohol.

`ALCDAY4` is present in the BRFSS 2024 dataset.

## Why I selected it

Alcohol consumption is an important behavioral characteristic.

It may interact with:

* cardiovascular health
* metabolic health
* lifestyle patterns
* blood pressure
* other risk behaviors

The goal is not to assume that alcohol automatically causes cardiac disease. It is included as a potentially useful behavioral predictor.

## ML type

Special encoded numerical/categorical variable.

## Suggested name

```text
alcohol_frequency
```

## Important cleaning issue

`ALCDAY4` uses special encoding.

Do not treat its raw numbers as an ordinary continuous measurement without consulting its codebook.

---

# 19. Predictor 14 — `_EDUCAG`

## Friendly name

**Education Level**

## Category

Socioeconomic

## What it represents

A calculated grouped education variable.

`_EDUCAG` is present in the calculated section of BRFSS 2024.

## Why I selected it

Education can be associated with:

* health literacy
* healthcare access
* occupation
* nutrition
* smoking behavior
* physical activity
* preventive health practices

Socioeconomic variables are important because cardiovascular health is influenced by more than biological factors alone.

## ML type

```text
Ordinal categorical
```

## Suggested name

```text
education_level
```

---

# 20. Predictor 15 — `INCOME3`

## Friendly name

**Household Income Category**

## Category

Socioeconomic

## What it represents

Reported household income category.

`INCOME3` is included directly in BRFSS 2024.

## Why I selected it

Income may influence:

* access to healthcare
* diet
* medication access
* preventive care
* living environment
* health behavior

It gives the model socioeconomic information that clinical variables alone cannot provide.

## ML type

```text
Ordinal categorical
```

## Suggested name

```text
income_level
```

## Important consideration

Income questions commonly have relatively high refusal or missing rates.

Therefore we should examine:

```python
df["INCOME3"].value_counts(dropna=False)
```

before deciding how to handle missing values.

---

# 21. Predictor 16 — `EMPLOY1`

## Friendly name

**Employment Status**

## Category

Socioeconomic

## What it represents

Current employment situation.

The variable is explicitly present in BRFSS 2024.

Possible categories can represent concepts such as:

```text
employed
self-employed
unemployed
retired
unable to work
student
homemaker
```

## Why I selected it

Employment status captures socioeconomic and lifestyle differences.

For example, it can indirectly relate to:

* age
* physical ability
* income
* healthcare access
* lifestyle

## ML type

```text
Nominal categorical
```

## Suggested name

```text
employment_status
```

---

# 22. Predictor 17 — `DIFFWALK`

## Friendly name

**Difficulty Walking or Climbing Stairs**

## Category

Functional health

## What it represents

Whether the respondent reports serious difficulty walking or climbing stairs.

`DIFFWALK` appears directly in the BRFSS 2024 dataset.

## Why I selected it

It provides information about physical function that BMI and general health do not fully capture.

Reduced physical ability may be associated with:

* chronic illness
* obesity
* cardiovascular limitations
* older age
* physical inactivity

## Important caution

This variable could sometimes represent a **consequence of existing illness**, rather than a factor that existed before cardiovascular disease.

That is acceptable for our current:

> disease-status classification

but it would be more questionable for:

> future disease prediction.

## ML type

```text
Categorical / binary
```

## Suggested name

```text
difficulty_walking
```

---

# 23. Predictor 18 — `PERSDOC3`

## Friendly name

**Has Personal Healthcare Provider**

## Category

Healthcare access

## What it represents

Information about whether a respondent has one or more people they consider their personal healthcare provider.

`PERSDOC3` is present in BRFSS 2024.

## Why I selected it

Having regular access to healthcare may affect:

* early diagnosis
* screening
* medication
* disease management
* preventive care

It provides healthcare-access information absent from biological features.

## ML type

```text
Categorical
```

## Suggested name

```text
personal_doctor
```

---

# 24. Predictor 19 — `MEDCOST1`

## Friendly name

**Could Not See Doctor Because of Cost**

## Category

Healthcare access / socioeconomic

## What it represents

Whether medical cost prevented the respondent from obtaining needed healthcare.

`MEDCOST1` is part of the BRFSS 2024 core dataset.

## Why I selected it

Healthcare affordability can influence:

* diagnosis
* medication adherence
* preventive care
* chronic-disease management
* medical follow-up

It captures an important dimension that neither income nor insurance necessarily captures completely.

Someone can have income but still face healthcare affordability problems.

## ML type

```text
Categorical / binary
```

## Suggested name

```text
medical_cost_barrier
```

---

# 25. Why These 19 Predictors Work Well Together

The purpose was not simply to select the 19 variables most correlated with `_MICHD`.

We want a **balanced representation of different types of information**.

## Demographics

```text
_AGE80
_SEX
```

These establish basic demographic risk context.

---

## Physical characteristics

```text
_BMI5
```

This captures body composition.

---

## Overall health

```text
GENHLTH
PHYSHLTH
MENTHLTH
```

These provide information about health burden and wellbeing.

---

## Medical conditions

```text
DIABETE4
CVDSTRK3
CHCKDNY2
```

These represent major comorbid health conditions.

---

## Lifestyle

```text
_TOTINDA
_SMOKER3
ALCDAY4
```

These represent modifiable behaviors.

---

## Socioeconomic factors

```text
_EDUCAG
INCOME3
EMPLOY1
```

These represent socioeconomic context.

---

## Functional health

```text
DIFFWALK
```

This captures physical functional limitation.

---

## Healthcare access

```text
CHECKUP1
PERSDOC3
MEDCOST1
```

These describe interaction with and access to healthcare.

---

# 26. Why This Is Better Than Selecting 20 Random Variables

Our feature design has approximately:

```text
Demographics           2
Physical                1
Health status           3
Medical conditions      3
Lifestyle               3
Socioeconomic           3
Functional              1
Healthcare access       3
--------------------------------
Predictors             19
Target                  1
--------------------------------
Total                  20
```

This gives the model diverse information without using hundreds of unnecessary variables.

---

# 27. Raw Dataset to Project Dataset

The workflow should be:

```text
BRFSS 2024
457,670 × 345
        │
        ▼
Select exactly 20 variables
        │
        ▼
457,670 × 20
        │
        ▼
Decode CDC special values
        │
        ▼
Remove invalid target records
        │
        ▼
Rename variables
        │
        ▼
Train / Validation / Test
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Machine Learning
```

---

# 28. Python Selection Code

```python
selected_columns = [
    "_AGE80",
    "_SEX",
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "CHECKUP1",
    "_TOTINDA",
    "DIABETE4",
    "CVDSTRK3",
    "CHCKDNY2",
    "_SMOKER3",
    "ALCDAY4",
    "_EDUCAG",
    "INCOME3",
    "EMPLOY1",
    "DIFFWALK",
    "PERSDOC3",
    "MEDCOST1",
    "_MICHD"
]

cardiac = df[selected_columns].copy()

print(cardiac.shape)
```

The number of rows initially remains approximately the same as BRFSS.

The number of columns should be:

```python
assert cardiac.shape[1] == 20
```

---

# 29. Recommended Renaming

Once the variables have been selected and decoded, rename them to make the project easier to understand.

```python
rename_map = {
    "_AGE80": "age",
    "_SEX": "sex",
    "_BMI5": "bmi",
    "GENHLTH": "general_health",
    "PHYSHLTH": "poor_physical_health_days",
    "MENTHLTH": "poor_mental_health_days",
    "CHECKUP1": "last_checkup",
    "_TOTINDA": "physical_activity",
    "DIABETE4": "diabetes_status",
    "CVDSTRK3": "stroke_history",
    "CHCKDNY2": "kidney_disease",
    "_SMOKER3": "smoking_status",
    "ALCDAY4": "alcohol_frequency",
    "_EDUCAG": "education_level",
    "INCOME3": "income_level",
    "EMPLOY1": "employment_status",
    "DIFFWALK": "difficulty_walking",
    "PERSDOC3": "personal_doctor",
    "MEDCOST1": "medical_cost_barrier",
    "_MICHD": "cardiac_disease"
}

cardiac = cardiac.rename(columns=rename_map)
```

Your final columns become:

```text
age
sex
bmi
general_health
poor_physical_health_days
poor_mental_health_days
last_checkup
physical_activity
diabetes_status
stroke_history
kidney_disease
smoking_status
alcohol_frequency
education_level
income_level
employment_status
difficulty_walking
personal_doctor
medical_cost_barrier
cardiac_disease
```

---

# 30. Feature / Target Separation

After cleaning:

```python
X = cardiac.drop(columns=["cardiac_disease"])

y = cardiac["cardiac_disease"]
```

Verify:

```python
print(X.shape)
print(y.shape)
```

You should have:

```text
X → 19 features

y → 1 target
```

---

# 31. Numerical Features

A reasonable starting classification is:

```python
numeric_features = [
    "age",
    "bmi",
    "poor_physical_health_days",
    "poor_mental_health_days"
]
```

These can use:

```text
Median imputation
+
StandardScaler
```

For example:

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

This is especially useful for:

* Logistic Regression
* Neural Networks
* KNN
* SVM

---

# 32. Categorical Features

The remaining predictors are primarily categorical:

```python
categorical_features = [
    "sex",
    "general_health",
    "last_checkup",
    "physical_activity",
    "diabetes_status",
    "stroke_history",
    "kidney_disease",
    "smoking_status",
    "alcohol_frequency",
    "education_level",
    "income_level",
    "employment_status",
    "difficulty_walking",
    "personal_doctor",
    "medical_cost_barrier"
]
```

A standard pipeline could be:

```python
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "one_hot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])
```

However, **CDC special codes must be decoded before this stage**.

---

# 33. Special Missing Values

This is extremely important.

BRFSS does not represent every missing value as:

```python
NaN
```

Survey variables may use values such as:

```text
7
9
77
88
99
777
999
```

for different meanings.

Depending on the particular variable, these may represent:

```text
Don't know
Refused
None
Not asked
Missing
```

CDC specifically warns that refusal and missing responses can use codes such as 9, 99, or 999 and that analysts must carefully handle these categories.

Therefore NEVER do:

```python
df.replace(9, np.nan)
```

globally.

Instead:

```text
Read codebook
       ↓
Determine codes for each variable
       ↓
Convert only appropriate codes
       ↓
NaN
```

---

# 34. Recommended Data Types

| Variable                    | ML Type                     |
| --------------------------- | --------------------------- |
| `age`                       | Numeric                     |
| `sex`                       | Categorical                 |
| `bmi`                       | Numeric                     |
| `general_health`            | Ordinal categorical         |
| `poor_physical_health_days` | Numeric                     |
| `poor_mental_health_days`   | Numeric                     |
| `last_checkup`              | Ordinal categorical         |
| `physical_activity`         | Binary categorical          |
| `diabetes_status`           | Categorical                 |
| `stroke_history`            | Binary categorical          |
| `kidney_disease`            | Binary categorical          |
| `smoking_status`            | Categorical                 |
| `alcohol_frequency`         | Encoded behavioral variable |
| `education_level`           | Ordinal categorical         |
| `income_level`              | Ordinal categorical         |
| `employment_status`         | Nominal categorical         |
| `difficulty_walking`        | Binary categorical          |
| `personal_doctor`           | Categorical                 |
| `medical_cost_barrier`      | Binary categorical          |
| `cardiac_disease`           | Binary target               |

---

# 35. Model Strategy

This 20-column dataset is suitable for both classical ML and deep learning.

Start with:

```text
Dummy Classifier
        ↓
Logistic Regression
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
Neural Network
```

Do not start with the neural network.

The Logistic Regression baseline gives you something interpretable against which the more complex models can be compared.

---

# 36. Why This Dataset Is Good for Neural Networks

You have approximately:

```text
457,670 observations
×
19 predictors
```

before exclusions for missing target values.

That is enough data to meaningfully demonstrate:

```text
Forward propagation

Loss calculation

Backpropagation

Gradient descent

Mini-batches

Epochs

Learning rates

SGD

Adam

RMSprop
```

For example:

```python
batch_size = 128
epochs = 30
```

and compare learning rates such as:

```python
1e-5
1e-3
1e-1
```

while plotting training and validation loss.

---

# 37. Recommended Neural Network

After encoding and scaling:

```python
model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])
```

For binary classification:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

# 38. Final Evaluation Metrics

Because cardiac disease will probably be less common than the negative class, do not rely exclusively on accuracy.

Report:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion Matrix
```

For this project, particularly pay attention to:

```text
Recall for cardiac_disease = 1
```

because false negatives are important when interpreting a health-related classification task.

---

# 39. Important Scientific Limitation

This dataset should **not** be described as predicting who will develop cardiovascular disease in the future.

BRFSS is primarily cross-sectional survey data.

Our target is:

> whether the respondent reports a history of coronary heart disease or myocardial infarction.

Therefore the project should be called something like:

# Cardiovascular Disease Classification Using Large-Scale Real-World Health Survey Data

rather than:

# 10-Year Heart Disease Prediction

The latter would require longitudinal outcome data.

---

# 40. Final Dataset Design

The final structure is:

```text
INPUT
│
├── Age
├── Sex
├── BMI
│
├── General Health
├── Physical Health
├── Mental Health
│
├── Physical Activity
├── Smoking
├── Alcohol
│
├── Diabetes
├── Stroke History
├── Kidney Disease
│
├── Education
├── Income
├── Employment
│
├── Difficulty Walking
│
├── Last Checkup
├── Personal Doctor
└── Medical Cost Barrier

             ↓

       Machine Learning

             ↓

cardiac_disease

0 → No reported CHD / MI

1 → Reported CHD / MI
```

---

# 41. Final Decision

These **20 columns** provide a strong balance between:

* medical history
* demographics
* lifestyle
* physical condition
* socioeconomic status
* healthcare access
* functional health

while avoiding direct target leakage.

The final dataset should therefore contain:

```python
[
    "_AGE80",
    "_SEX",
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "CHECKUP1",
    "_TOTINDA",
    "DIABETE4",
    "CVDSTRK3",
    "CHCKDNY2",
    "_SMOKER3",
    "ALCDAY4",
    "_EDUCAG",
    "INCOME3",
    "EMPLOY1",
    "DIFFWALK",
    "PERSDOC3",
    "MEDCOST1",
    "_MICHD"
]
```

That gives:

**19 carefully selected predictors + 1 binary cardiac target = exactly 20 columns.**
