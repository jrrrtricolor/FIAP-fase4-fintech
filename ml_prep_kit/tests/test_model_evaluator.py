import json
import unittest

import numpy as np
import pandas as pd

from ml_prep_kit import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    def test_evaluate_classification_returns_expected_metrics(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([0, 1, 1, 1])
        y_score = pd.Series([0.1, 0.6, 0.8, 0.9])

        metrics = evaluator.evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIn("roc_auc", metrics)

    def test_evaluate_multiclass_returns_macro_and_per_class_metrics(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series(
            ["classe_a", "classe_a", "classe_b", "classe_b", "classe_c", "classe_c"]
        )
        y_pred = pd.Series(
            ["classe_a", "classe_b", "classe_b", "classe_b", "classe_c", "classe_a"]
        )

        metrics = evaluator.evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
        )
        class_metrics = evaluator.evaluate_classification_by_class(
            y_true=y_true,
            y_pred=y_pred,
        )

        self.assertIn("f1_macro", metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertEqual(
            set(class_metrics["class"]),
            {"classe_a", "classe_b", "classe_c"},
        )
        self.assertEqual(
            set(class_metrics.columns),
            {"class", "precision", "recall", "f1", "support"},
        )

    def test_binary_report_has_correct_metrics_and_class_orientation(self):
        report = ModelEvaluator().evaluate_binary_classification(
            [0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9]
        )
        metrics = report["metrics"]
        np.testing.assert_array_equal(
            report["confusion_matrix"].to_numpy(), [[1, 1], [0, 2]]
        )
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["false_negative_rate"], 0.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.105)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(report["by_class"]["support"].tolist(), [2, 2])
        self.assertEqual(report["calibration"]["support"].sum(), 4)
        self.assertEqual(report["roc_curve"].iloc[-1]["true_positive_rate"], 1.0)
        self.assertEqual(report["precision_recall_curve"].iloc[-1]["recall"], 0.0)

    def test_binary_report_applies_threshold_without_changing_ranking_metrics(self):
        evaluator = ModelEvaluator()
        lower = evaluator.evaluate_binary_classification(
            [0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9]
        )
        higher = evaluator.evaluate_binary_classification(
            [0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9], threshold=0.85
        )
        np.testing.assert_array_equal(
            higher["confusion_matrix"].to_numpy(), [[2, 0], [1, 1]]
        )
        self.assertEqual(higher["metrics"]["precision"], 1.0)
        self.assertEqual(higher["metrics"]["recall"], 0.5)
        for name in ["roc_auc", "average_precision", "brier_score", "log_loss"]:
            self.assertEqual(lower["metrics"][name], higher["metrics"][name])

    def test_binary_report_handles_single_class_and_undefined_rates(self):
        report = ModelEvaluator().evaluate_binary_classification([0, 0], [0.1, 0.2])
        metrics = report["metrics"]
        for name in ["roc_auc", "average_precision", "recall", "precision", "f1"]:
            self.assertIsNone(metrics[name])
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(report["by_class"]["class"].tolist(), [0, 1])
        self.assertEqual(report["by_class"]["support"].tolist(), [2, 0])
        self.assertTrue(report["roc_curve"].empty)
        self.assertTrue(report["precision_recall_curve"].empty)
        self.assertTrue(report["notes"])
        json.dumps(metrics, allow_nan=False)

    def test_binary_report_accepts_probability_endpoints(self):
        report = ModelEvaluator().evaluate_binary_classification([0, 1], [0.0, 1.0])
        self.assertEqual(report["metrics"]["brier_score"], 0.0)
        self.assertEqual(report["calibration"]["bin"].tolist(), [0, 9])
        self.assertTrue(np.isfinite(report["metrics"]["log_loss"]))

    def test_binary_report_rejects_invalid_inputs(self):
        cases = [
            ([], [], 0.5),
            ([0, 1], [0.1], 0.5),
            ([0, 2], [0.1, 0.9], 0.5),
            ([0, 1], [[0.9, 0.1], [0.1, 0.9]], 0.5),
            ([0, 1], [np.nan, 0.9], 0.5),
            ([0, 1], [np.inf, 0.9], 0.5),
            ([0, 1], [-0.1, 0.9], 0.5),
            ([0, 1], [0.1, 1.1], 0.5),
            ([0, 1], [0.1, 0.9], -0.1),
            ([0, 1], [0.1, 0.9], 1.1),
            ([0, 1], [0.1, 0.9], np.nan),
        ]
        for target, scores, threshold in cases:
            with self.subTest(target=target, scores=scores, threshold=threshold):
                with self.assertRaises(ValueError):
                    ModelEvaluator().evaluate_binary_classification(
                        target, scores, threshold=threshold
                    )

    def test_bootstrap_is_reproducible_and_does_not_mutate_inputs(self):
        evaluator = ModelEvaluator()
        target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.3, 0.4, 0.6, 0.4, 0.7, 0.8, 0.9])
        original_scores = scores.copy()
        first = evaluator.bootstrap_binary_metrics(target, scores, n_resamples=20)
        second = evaluator.bootstrap_binary_metrics(target, scores, n_resamples=20)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_array_equal(scores, original_scores)
        valid_counts = first.set_index("metric")["valid_resamples"]
        self.assertTrue(valid_counts.drop("precision").eq(20).all())
        self.assertGreater(valid_counts["precision"], 0)
        self.assertLess(valid_counts["precision"], 20)
        self.assertTrue((first["lower"] <= first["upper"]).all())
        self.assertTrue(first["lower"].between(0, 1).all())
        self.assertTrue(first["upper"].between(0, 1).all())

    def test_bootstrap_requires_two_classes_and_valid_configuration(self):
        evaluator = ModelEvaluator()
        with self.assertRaisesRegex(ValueError, "duas classes"):
            evaluator.bootstrap_binary_metrics([0, 0], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "n_resamples"):
            evaluator.bootstrap_binary_metrics([0, 1], [0.1, 0.9], n_resamples=1)
        with self.assertRaisesRegex(ValueError, "confidence_level"):
            evaluator.bootstrap_binary_metrics([0, 1], [0.1, 0.9], confidence_level=1.0)


if __name__ == "__main__":
    unittest.main()
