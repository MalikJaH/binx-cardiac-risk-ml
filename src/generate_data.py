"""Generate a deterministic, privacy-safe synthetic cardiac dataset.

The deliberately injected quality issues (missing values, implausible values,
and duplicate rows) make the cleaning section of the project reproducible.
This dataset is educational only and must not be used for clinical decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_PATIENTS = 1_200
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "cardio_patients_raw.csv"


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert log-odds to probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def generate_dataset() -> pd.DataFrame:
    """Return synthetic cardiac observations with realistic relationships."""
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(30, 86, N_PATIENTS)
    sex = rng.choice(["Female", "Male"], N_PATIENTS, p=[0.48, 0.52])
    resting_bp = np.clip(rng.normal(126 + 0.28 * (age - 50), 15), 85, 205)
    cholesterol = np.clip(rng.normal(205 + 0.55 * (age - 50), 38), 105, 390)
    bmi = np.clip(rng.normal(27.0 + 0.035 * (age - 50), 4.2), 16, 44)
    fasting_blood_sugar = rng.binomial(1, sigmoid(-2.0 + 0.045 * (age - 50) + 0.10 * (bmi - 25)))
    max_heart_rate = np.clip(rng.normal(181 - 0.72 * age, 13), 70, 205)
    exercise_angina = np.where(
        rng.random(N_PATIENTS) < sigmoid(-1.4 + 0.040 * (age - 50) + 0.055 * (bmi - 25)),
        "Yes",
        "No",
    )
    chest_pain_type = rng.choice(
        ["Typical", "Atypical", "Non-anginal", "Asymptomatic"],
        N_PATIENTS,
        p=[0.17, 0.25, 0.30, 0.28],
    )

    chest_pain_effect = pd.Series(chest_pain_type).map(
        {"Typical": 0.25, "Atypical": 0.0, "Non-anginal": -0.25, "Asymptomatic": 0.90}
    ).to_numpy()
    log_odds = (
        -1.65
        + 0.045 * (age - 50)
        + 0.30 * (sex == "Male")
        + 0.020 * (resting_bp - 120)
        + 0.008 * (cholesterol - 200)
        + 0.090 * (bmi - 25)
        + 0.45 * fasting_blood_sugar
        - 0.018 * (max_heart_rate - 145)
        + 0.95 * (exercise_angina == "Yes")
        + chest_pain_effect
        + rng.normal(0, 0.45, N_PATIENTS)
    )
    cardiac_risk = rng.binomial(1, sigmoid(log_odds))

    data = pd.DataFrame(
        {
            "patient_id": [f"SYN-{value:04d}" for value in range(1, N_PATIENTS + 1)],
            "age": age.astype(float),
            "sex": sex,
            "resting_bp": resting_bp.round(1),
            "cholesterol": cholesterol.round(1),
            "bmi": bmi.round(1),
            "fasting_blood_sugar": fasting_blood_sugar,
            "max_heart_rate": max_heart_rate.round(1),
            "exercise_angina": exercise_angina,
            "chest_pain_type": chest_pain_type,
            "cardiac_risk": cardiac_risk,
        }
    )

    # Inject missing values without changing the already-generated target.
    for column in ["resting_bp", "cholesterol", "bmi", "max_heart_rate", "chest_pain_type"]:
        missing_rows = rng.choice(data.index, size=round(0.025 * N_PATIENTS), replace=False)
        data.loc[missing_rows, column] = np.nan

    # Inject implausible values that the notebook must detect and convert to missing.
    invalid_values = {
        "age": [150.0, 7.0],
        "resting_bp": [25.0, 310.0],
        "cholesterol": [-15.0, 850.0],
        "bmi": [7.0, 75.0],
        "max_heart_rate": [35.0, 290.0],
    }
    available = data.index.to_numpy().copy()
    rng.shuffle(available)
    cursor = 0
    for column, values in invalid_values.items():
        for value in values:
            data.loc[available[cursor], column] = value
            cursor += 1

    data.loc[available[cursor], "chest_pain_type"] = "Unknown"

    # Exact duplicates are retained so the cleaning audit has evidence to report.
    duplicates = data.sample(15, random_state=RANDOM_STATE)
    data = pd.concat([data, duplicates], ignore_index=True)
    return data


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset):,} rows to {OUTPUT_PATH}")
    print(f"Target prevalence: {dataset['cardiac_risk'].mean():.1%}")


if __name__ == "__main__":
    main()

