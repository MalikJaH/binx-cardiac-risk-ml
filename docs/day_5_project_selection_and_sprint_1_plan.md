# Day 5 — Phase 3 Project Selection & Sprint 1 Planning

**Project:** Cardiac Patient Monitoring — Synthetic Cardiac-Risk Classification  
**Phase:** Phase 3 Capstone  
**Planned duration:** Four one-week sprints (Weeks 6–9)  
**Day 5 allocation:** 8 hours  
**Status:** Ready to begin Sprint 1

## 1. Selected capstone project

### Selected option

**Comparable live BinX Tech use case: binary classification.** The project predicts whether a synthetic patient record shows an **elevated simulated cardiac-risk pattern** (`cardiac_risk = 1`) or a lower pattern (`cardiac_risk = 0`).

This selection follows the same end-to-end workflow as the programme's classification options: data selection and validation, exploratory data analysis (EDA), baseline modelling, model comparison, evaluation, and deployment as a usable educational demonstration.
## 2. Problem statement

Can routinely shaped **synthetic** cardiac attributes classify an elevated simulated cardiac-risk pattern versus a lower pattern?

The model will use validated predictor features such as age, resting blood pressure, cholesterol, BMI, fasting blood-sugar indicator, maximum heart rate, exercise angina, and chest-pain type. `patient_id` will be excluded from modelling. Success is measured as an improvement over a transparent Logistic Regression baseline using cross-validated ROC-AUC, with precision, recall, F1, and a held-out test set reported for context.

## 3. Definition of Done

The capstone is complete only when all of the following are true:

- A clean, documented Jupyter Notebook covers the complete pipeline: data audit and EDA, preprocessing, modelling, evaluation, and conclusions.
- A trained model is saved with reproducible preprocessing and reported metrics, including a comparison with the baseline.
- A usable Streamlit or FastAPI deployment is available at a public URL and clearly labels the project as educational and non-clinical.
- The GitHub repository has a clear README, `requirements.txt`, source code, notebook, model artifact, and reproducible run instructions.
- A short technical write-up explains the problem, data, approach, results, limitations, and responsible-use constraints.
- Work is performed through feature branches and reviewed through pull requests before merge.

### Visible delivery checklist

| Deliverable | Target evidence |
| --- | --- |
| Data and EDA | Data dictionary, validation audit, descriptive analysis, and charts |
| Baseline and candidate models | Executed notebook, cross-validation table, and saved pipeline |
| Final evaluation | Held-out metrics, confusion matrix, ROC curve, and error discussion |
| Deployment | Public Streamlit/FastAPI URL with educational-use disclaimer |
| Documentation | README, technical write-up, requirements, and demo instructions |
| Engineering workflow | Feature branches, meaningful commits, and reviewed pull requests |

## 4. Sprint 1 goal

**Understand the synthetic dataset and establish a reproducible baseline model that future sprint work must beat.**

Sprint 1 ends with a validated dataset, documented EDA findings, a leakage-safe Logistic Regression baseline, and an agreed metric benchmark. It does not claim clinical validity or production readiness.

## 5. Sprint 1 backlog

| ID | Backlog item | Estimate | Priority |
| --- | --- | ---: | --- |
| S1-01 | Confirm scope, problem statement, Definition of Done, and responsible-use notice | 1 hour | Must have |
| S1-02 | Review dataset, data dictionary, quality issues, and validation rules | 2 hours | Must have |
| S1-03 | Complete and document EDA | 2 hours | Must have |
| S1-04 | Build a leakage-safe baseline preprocessing and Logistic Regression pipeline | 2 hours | Must have |
| S1-05 | Record baseline metrics, update documentation, and open a pull request | 1 hour | Must have |
|  | **Total planned effort** | **8 hours** |  |  |

## 6. Acceptance criteria

### S1-01 — Scope and Definition of Done

- [ ] The problem statement identifies the target, intended educational purpose, and non-clinical limitation.
- [ ] The Definition of Done is recorded in this document and linked from the project README or issue tracker.
- [ ] The project scope is confirmed before Phase 3 implementation begins.
- [ ] Changes are made on a feature branch and submitted in a pull request.

### S1-02 — Dataset selection and quality audit

- [ ] The selected dataset and its synthetic-data limitation are documented.
- [ ] Feature meanings, target definition, and expected ranges are documented in `data/data_dictionary.md`.
- [ ] Duplicates, missing values, invalid values, invalid categories, and target balance are checked and their handling is recorded in Markdown.
- [ ] The notebook runs without errors from a clean environment.
- [ ] The work is committed with a clear message and opened as a pull request before merging.

### S1-03 — Exploratory data analysis

- [ ] EDA includes descriptive statistics, distributions, class balance, relevant relationship plots, and correlation analysis.
- [ ] At least three data-informed findings are written in Markdown, including limitations of the synthetic data.
- [ ] Charts and summary outputs are reproducible from the notebook.
- [ ] No modelling decision uses the held-out test set.
- [ ] The completed work is committed on the correct feature branch and reviewed through a pull request.

### S1-04 — Baseline model

- [ ] The data is split into stratified training and held-out test sets with a recorded random seed.
- [ ] Preprocessing is fitted only on training data and is contained in a reusable scikit-learn pipeline.
- [ ] A Logistic Regression baseline is trained and evaluated with stratified cross-validation.
- [ ] ROC-AUC, precision, recall, F1, and accuracy are logged; the model-selection metric is explicitly identified as ROC-AUC.
- [ ] The held-out test set is reserved for final reporting, not used to tune or select the baseline.
- [ ] The notebook executes without errors and results are committed through a pull request.

### S1-05 — Baseline report and review

- [ ] Baseline metrics are written to a versioned CSV or Markdown summary and linked from the README.
- [ ] The Sprint 1 result states the benchmark future models must beat and any notable data-quality risks.
- [ ] A pull request summarises completed tasks, evidence, and any follow-up work for Sprint 2.
- [ ] The pull request is reviewed before merge.

## 7. GitHub branch and pull-request workflow

1. Create an issue or backlog item for the task.
2. Branch from the `main` branch using `feature/sprint-1-baseline`.
3. Make focused changes, run the notebook or relevant checks, and commit with a clear message such as `feat: add leakage-safe logistic regression baseline`.
4. Push the branch and open a pull request that links the backlog item, lists validation performed, and includes relevant metrics or screenshots.
5. Request a pull-request review. Address comments and keep the PR updated.
6. Merge only after approval, then update the sprint backlog and project documentation.
