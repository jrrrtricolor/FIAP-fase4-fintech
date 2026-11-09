from pathlib import Path

from ml_prep_kit import (
    CSVDataLoader,
    HomeCreditData,
)


def main():
    """Reutiliza a referência e o gerador de drift do framework.

    Exemplo:
        python -m src.credit_monitoring.generate_drift_data
    """
    project_root = Path.cwd()
    processed_path = project_root / "data/processed"
    preparation = HomeCreditData()
    experiment = preparation.load_experiment(processed_path)
    preparation.random_state = experiment["random_state"]
    version_one = CSVDataLoader(processed_path).load({"v1": "credit_scoring.csv"})["v1"]

    preparation.create_drift_version(
        version_one,
        income_factor=0.70,
        credit_factor=1.40,
        output_path=processed_path / "credit_scoring_v2.csv",
    )


if __name__ == "__main__":
    main()
