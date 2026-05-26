import unittest

import numpy as np
import pandas as pd

from src.offline_policy_improvement import (
    add_action_value_features,
    add_policy_gate_columns,
    add_shape_risk_features,
    add_submission_price_features,
    rank_policy_candidates,
    baseline_windows_from_submission_frame,
    generate_nearby_actions,
)


class OfflinePolicyImprovementTests(unittest.TestCase):
    def test_generate_nearby_actions_keeps_feasible_nonbaseline_windows(self):
        baseline = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "baseline_charge_start": 10,
                    "baseline_discharge_start": 20,
                }
            ]
        )

        actions = generate_nearby_actions(baseline, max_shift=1)

        self.assertFalse(actions.empty)
        self.assertFalse(
            (
                actions["candidate_charge_start"].eq(10)
                & actions["candidate_discharge_start"].eq(20)
            ).any()
        )
        self.assertTrue((actions["candidate_discharge_start"] >= actions["candidate_charge_start"] + 8).all())
        self.assertLessEqual(actions["max_abs_start_delta"].max(), 1)

    def test_add_action_value_features_computes_seed_delta_and_true_delta(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        price_df = pd.DataFrame(
            {
                "times": times,
                "pred_price_seed42": np.arange(96, dtype=float),
                "pred_price_seed2024": np.arange(96, dtype=float),
                "A": np.arange(96, dtype=float),
            }
        )
        actions = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "baseline_charge_start": 10,
                    "baseline_discharge_start": 20,
                    "candidate_charge_start": 8,
                    "candidate_discharge_start": 25,
                    "delta_charge_start": -2,
                    "delta_discharge_start": 5,
                    "max_abs_start_delta": 5,
                }
            ]
        )

        out = add_action_value_features(
            actions,
            price_df,
            seed_cols=["pred_price_seed42", "pred_price_seed2024"],
            true_col="A",
            power_value=1.0,
        )

        self.assertIn("pred_seed_delta_min", out.columns)
        self.assertIn("true_delta_profit", out.columns)
        self.assertGreater(out.loc[0, "pred_seed_delta_min"], 0.0)
        self.assertAlmostEqual(out.loc[0, "true_delta_profit"], out.loc[0, "pred_seed_delta_mean"])

    def test_submission_price_features_catch_opposite_delta_direction(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        prices = np.zeros(96, dtype=float)
        prices[10:18] = 1.0
        prices[20:28] = 5.0
        prices[8:16] = 2.0
        prices[25:33] = 4.0
        reference = pd.DataFrame({"times": times, "实时价格": prices, "power": np.zeros(96)})
        actions = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "baseline_charge_start": 10,
                    "baseline_discharge_start": 20,
                    "candidate_charge_start": 8,
                    "candidate_discharge_start": 25,
                    "pred_seed_delta_min": 100.0,
                }
            ]
        )

        out = add_submission_price_features(actions, reference, power_value=1.0)

        self.assertLess(out.loc[0, "submission_price_delta"], 0.0)
        self.assertFalse(out.loc[0, "multi_price_delta_agree"])

    def test_shape_risk_features_and_gate_reject_bad_tail(self):
        candidates = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "delta_charge_start": -1,
                    "delta_discharge_start": 2,
                    "offline_pred_delta_mean": 200.0,
                    "offline_pred_delta_std": 20.0,
                    "pred_seed_delta_min": 150.0,
                    "pred_seed_delta_positive_count": 3,
                    "max_abs_start_delta": 2,
                    "total_abs_start_delta": 3,
                    "submission_price_delta": 10.0,
                },
                {
                    "date": "2026-01-02",
                    "delta_charge_start": 1,
                    "delta_discharge_start": 1,
                    "offline_pred_delta_mean": 200.0,
                    "offline_pred_delta_std": 20.0,
                    "pred_seed_delta_min": 150.0,
                    "pred_seed_delta_positive_count": 3,
                    "max_abs_start_delta": 1,
                    "total_abs_start_delta": 2,
                    "submission_price_delta": 10.0,
                },
            ]
        )
        historical = pd.DataFrame(
            [
                {"delta_charge_start": -1, "delta_discharge_start": 2, "true_delta_profit": -100.0},
                {"delta_charge_start": -1, "delta_discharge_start": 2, "true_delta_profit": 50.0},
                {"delta_charge_start": 1, "delta_discharge_start": 1, "true_delta_profit": 10.0},
                {"delta_charge_start": 1, "delta_discharge_start": 1, "true_delta_profit": 20.0},
            ]
        )

        enriched = add_shape_risk_features(candidates, historical, quantile=0.10)
        gated = add_policy_gate_columns(
            enriched,
            lower_confidence_lambda=1.0,
            min_offline_delta_lower=100.0,
            min_pred_seed_delta=100.0,
            min_seed_positive_count=3,
            min_submission_price_delta=0.0,
            min_shape_sample_count=2,
            min_shape_positive_rate=0.75,
            min_shape_p10_delta=0.0,
        )

        self.assertFalse(gated.loc[0, "passes_policy_gate"])
        self.assertTrue(gated.loc[1, "passes_policy_gate"])
        self.assertLess(gated.loc[0, "shape_true_delta_p10"], 0.0)

    def test_baseline_windows_from_submission_frame_extracts_daily_starts(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        submission = pd.DataFrame({"times": times, "power": np.zeros(96)})
        submission.loc[10:17, "power"] = -1000.0
        submission.loc[25:32, "power"] = 1000.0

        windows = baseline_windows_from_submission_frame(submission)

        self.assertEqual(windows.iloc[0]["date"], "2026-01-01")
        self.assertEqual(windows.iloc[0]["baseline_charge_start"], 10)
        self.assertEqual(windows.iloc[0]["baseline_discharge_start"], 25)

    def test_rank_policy_candidates_filters_by_conservative_lower_bound(self):
        candidates = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "offline_pred_delta_mean": 200.0,
                    "offline_pred_delta_std": 20.0,
                    "pred_seed_delta_min": 150.0,
                    "pred_seed_delta_positive_count": 3,
                    "max_abs_start_delta": 1,
                    "total_abs_start_delta": 1,
                },
                {
                    "date": "2026-01-02",
                    "offline_pred_delta_mean": 250.0,
                    "offline_pred_delta_std": 300.0,
                    "pred_seed_delta_min": 200.0,
                    "pred_seed_delta_positive_count": 3,
                    "max_abs_start_delta": 1,
                    "total_abs_start_delta": 1,
                },
            ]
        )

        ranked = rank_policy_candidates(
            candidates,
            lower_confidence_lambda=1.0,
            min_offline_delta_lower=100.0,
            min_pred_seed_delta=100.0,
            min_seed_positive_count=3,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked.iloc[0]["date"], "2026-01-01")
        self.assertEqual(ranked.iloc[0]["offline_pred_delta_lower"], 180.0)
        self.assertTrue(ranked.iloc[0]["passes_policy_gate"])

    def test_policy_gate_columns_keep_rejected_rows_for_diagnostics(self):
        candidates = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "offline_pred_delta_mean": 200.0,
                    "offline_pred_delta_std": 20.0,
                    "pred_seed_delta_min": 150.0,
                    "pred_seed_delta_positive_count": 3,
                    "max_abs_start_delta": 1,
                    "total_abs_start_delta": 1,
                    "delta_gap_slots": 1,
                    "delta_charge_start": -1,
                    "delta_discharge_start": 1,
                    "submission_price_delta": 10.0,
                },
                {
                    "date": "2026-01-02",
                    "offline_pred_delta_mean": 200.0,
                    "offline_pred_delta_std": 20.0,
                    "pred_seed_delta_min": 80.0,
                    "pred_seed_delta_positive_count": 3,
                    "max_abs_start_delta": 1,
                    "total_abs_start_delta": 1,
                    "delta_gap_slots": -1,
                    "delta_charge_start": 1,
                    "delta_discharge_start": -1,
                    "submission_price_delta": -10.0,
                },
            ]
        )

        gated = add_policy_gate_columns(
            candidates,
            lower_confidence_lambda=1.0,
            min_offline_delta_lower=100.0,
            min_pred_seed_delta=100.0,
            min_seed_positive_count=3,
        )

        self.assertEqual(len(gated), 2)
        self.assertTrue(gated.loc[0, "passes_policy_gate"])
        self.assertFalse(gated.loc[1, "passes_policy_gate"])

        shape_gated = add_policy_gate_columns(
            candidates,
            lower_confidence_lambda=1.0,
            min_offline_delta_lower=100.0,
            min_pred_seed_delta=50.0,
            min_seed_positive_count=3,
            min_delta_gap_slots=0,
            forbid_charge_later_discharge_earlier=True,
        )
        self.assertTrue(shape_gated.loc[0, "passes_policy_gate"])
        self.assertFalse(shape_gated.loc[1, "passes_policy_gate"])

        price_gated = add_policy_gate_columns(
            candidates,
            lower_confidence_lambda=1.0,
            min_offline_delta_lower=100.0,
            min_pred_seed_delta=50.0,
            min_seed_positive_count=3,
            min_submission_price_delta=0.0,
        )
        self.assertTrue(price_gated.loc[0, "passes_policy_gate"])
        self.assertFalse(price_gated.loc[1, "passes_policy_gate"])


if __name__ == "__main__":
    unittest.main()
