from pathlib import Path

from ml_prep_kit import (
    HomeCreditTrainingWorkflow,
    ModelEvaluator,
)

from .api_constants import MLFLOW_MODEL_NAME


def main():
    project_root = Path.cwd()
    raw_path = project_root / "data/raw/home-credit-default-risk"
    processed_path = project_root / "data/processed"
    workflow = HomeCreditTrainingWorkflow(
        raw_path,
        processed_path,
        "tech-challenge-4-fintech-training",
        sample_size=20_000,
        random_state=42,
    )
    evaluator = ModelEvaluator()

    clean_df = workflow.load_data()
    _, datasets = workflow.split_data(clean_df)
    X_validation, y_validation = datasets["validacao"]

    tracker = workflow.tracker
    models, training_records = workflow.train_models()

    baseline_record = training_records["logistic_regression"]

    validation = evaluator.evaluate_models(
        models,
        {"validacao": (X_validation, y_validation)},
        thresholds={"dummy": 0.5, "logistic_regression": 0.5},
    )
    selected_threshold, threshold_results = evaluator.select_threshold(
        y_validation, validation["scores"]["logistic_regression"]["validacao"]
    )
    thresholds = {"dummy": 0.5, "logistic_regression": selected_threshold}
    evaluation = evaluator.evaluate_models(models, datasets, thresholds)
    confidence_intervals = evaluator.bootstrap_binary_metrics(
        datasets["teste"][1],
        evaluation["scores"]["logistic_regression"]["teste"],
        threshold=selected_threshold,
        n_resamples=500,
        confidence_level=0.95,
        random_state=workflow.preparation.random_state,
    )
    evaluation_path = evaluator.save_evaluation(
        evaluation,
        confidence_intervals,
        threshold_results,
        figures=None,
        directory=processed_path / "model/evaluation",
    )
    for name, reports in evaluation["reports"].items():
        tracker.log_evaluation(
            training_records[name]["run_id"],
            reports,
            artifacts=[evaluation_path],
            parameters={"threshold": thresholds[name]},
        )

    workflow.save_experiment(
        metadata={
            "threshold": selected_threshold,
            "training_run_id": baseline_record["run_id"],
            "model_uri": baseline_record["model_uri"],
        },
    )

    tracker.register_model(baseline_record["model_uri"], MLFLOW_MODEL_NAME)
    tracker.promote_latest_model_version(
        registered_model_name=MLFLOW_MODEL_NAME
    )

if __name__ == "__main__":
    main()
