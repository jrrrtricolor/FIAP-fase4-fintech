import sys
from datetime import UTC, datetime
from pathlib import Path

from ml_prep_kit import (
    CSVDataLoader,
    HomeCreditData,
)

def main():
    project_root = Path.cwd()
    processed_path = project_root / "data/processed"
    preparation = HomeCreditData()
    experiment = preparation.load_experiment(processed_path)
    preparation.random_state = experiment["random_state"]
    version_one = CSVDataLoader(processed_path).load({"v1": "credit_scoring.csv"})["v1"]

    income_factor, credit_factor = 0.70, 1.40
    version_two_path = processed_path / "credit_scoring_v2.csv"
    preparation.create_drift_version(
        version_one,
        income_factor=income_factor,
        credit_factor=credit_factor,
        output_path=version_two_path,
    )


if __name__ == "__main__":
    main()
