# Synthetic Cardiac Dataset — Data Dictionary

**Purpose:** Educational binary-classification capstone dataset.  
**Rows:** 100,000  
**Target:** `cardiac_risk`  
**Important:** Every record is synthetic. The dataset is not clinical data and must not be used for medical diagnosis or patient care.

| Column | Type | Description | Expected / clean range |
|---|---|---|---|
| `patient_id` | text | Synthetic record identifier; exclude from modelling | `SYN-000001`, ... |
| `age` | numeric | Age in years | 18–90 |
| `sex` | category | Synthetic sex category | `Female`, `Male` |
| `resting_bp` | numeric | Synthetic resting systolic blood pressure (mmHg) | approx. 85–210 |
| `cholesterol` | numeric | Synthetic serum cholesterol (mg/dL) | approx. 100–430 |
| `bmi` | numeric | Synthetic body mass index (kg/m²) | approx. 16–48 |
| `fasting_blood_sugar` | binary | Elevated fasting blood-sugar indicator | 0 or 1 |
| `max_heart_rate` | numeric | Synthetic maximum heart rate | approx. 65–205 |
| `smoking` | binary | Current-smoking indicator | 0 or 1 |
| `diabetes` | binary | Synthetic diabetes indicator | 0 or 1 |
| `physical_activity` | binary | Regular-physical-activity indicator | 0 or 1 |
| `family_history` | binary | Synthetic family-history indicator | 0 or 1 |
| `chest_pain_type` | category | Synthetic chest-pain category | `asymptomatic`, `non_anginal`, `atypical_angina`, `typical_angina` |
| `resting_ecg` | category | Synthetic resting-ECG category | `normal`, `st_t_abnormality`, `left_ventricular_hypertrophy` |
| `exercise_angina` | binary | Synthetic exercise-induced-angina indicator | 0 or 1 |
| `cardiac_risk` | binary target | Elevated simulated cardiac-risk pattern | 0 or 1 |

## Intentional data-quality issues

The raw dataset intentionally contains a small amount of:
- missing data in selected predictors;
- exact duplicate rows;
- implausible values in `age`, `resting_bp`, `cholesterol`, and `bmi`.

These issues are included so Sprint 1 can demonstrate data audit, validation, cleaning, EDA, and leakage-safe preprocessing.

## Target-generation note

`cardiac_risk` is generated probabilistically from several predictors plus latent noise. It is **not** a deterministic formula and no single feature directly determines the label. The positive-class prevalence is designed to be approximately 30%.
