from pathlib import Path

from ml_prep_kit import (
    CSVDataLoader,
    ExperimentTracker,
    FeaturePreprocessor,
    HomeCreditData,
    ModelEvaluator,
    ModelFactory,
)

from .api_constants import MLFLOW_MODEL_NAME


def main():
    project_root = Path.cwd()
    raw_path = project_root / "data/raw/home-credit-default-risk"
    processed_path = project_root / "data/processed"
    preparation = HomeCreditData(sample_size=20_000, random_state=42)
    evaluator = ModelEvaluator()

    raw_df = CSVDataLoader(raw_path).load(
        {"solicitacoes": "application_train.csv"},
        read_csv_kwargs={"usecols": preparation.source_columns},
    )["solicitacoes"]
    clean_df = preparation.prepare(preparation.sample(raw_df))

    partitions = preparation.split(clean_df)
    datasets = preparation.model_datasets(partitions)
    X_train, y_train = datasets["treino"]
    X_validation, y_validation = datasets["validacao"]

    factory = ModelFactory(random_state=preparation.random_state)
    model_parameters = {
        "dummy": {"strategy": "prior"},
        "logistic_regression": {"max_iter": 2_000, "class_weight": "balanced"},
    }
    tracker = ExperimentTracker.from_directory(
        "tech-challenge-4-fintech-training",
        processed_path / "monitoring",
        "training_artifacts",
    )
    tracking_uri = tracker.tracking_uri
    source_hash = preparation.source_hash(raw_path / "application_train.csv")
    preprocessor = FeaturePreprocessor(
        numeric_columns=preparation.numeric_features,
        categorical_columns=preparation.categorical_features,
        binary_columns=preparation.binary_features,
    )

    models, training_records = {}, {}
    for name, parameters in model_parameters.items():
        models[name] = factory.create_pipeline(
            preprocessor=preprocessor.create_pipeline(),
            problem_type="classification",
            model_name=name,
            parameters=parameters,
        )
        training_records[name] = tracker.train_model(
            name,
            models[name],
            X_train,
            y_train,
            parameters={
                **parameters,
                "model": name,
                "seed": preparation.random_state,
                "train_rows": len(X_train),
                "sample_size": preparation.sample_size,
            },
            tags={"source_sha256": source_hash},
        )

    baseline_record = training_records["logistic_regression"]

    validation = evaluator.evaluate_models(
        models,
        {"validacao": (X_validation, y_validation)},
        thresholds={"dummy": 0.5, "logistic_regression": 0.5},
    )
    selected_threshold, threshold_results = evaluator.select_threshold(
        y_validation, validation["scores"]["logistic_regression"]["validacao"]
    )

    preparation.save_experiment(
        processed_path,
        partitions,
        metadata={
            "threshold": selected_threshold,
            "tracking_uri": tracking_uri,
            "training_run_id": baseline_record["run_id"],
            "model_uri": baseline_record["model_uri"],
            "source_sha256": source_hash,
            "random_state": preparation.random_state,
            "sample_size": preparation.sample_size,
        },
    )

    tracker.register_model(baseline_record["model_uri"], MLFLOW_MODEL_NAME)
    tracker.promote_latest_model_version(
        registered_model_name=MLFLOW_MODEL_NAME
    )

if __name__ == "__main__":
    main()
