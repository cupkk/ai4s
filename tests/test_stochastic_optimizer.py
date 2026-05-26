import unittest

import numpy as np
import pandas as pd

from src.stochastic_optimizer import (
    detect_scenario_columns,
    optimize_one_day_scenarios,
)
from src.stochastic_candidate_pool import (
    add_seed_delta_diagnostics,
    apply_single_day_action,
    build_candidate_pool,
    default_scenario_sets,
    filter_candidate_pool,
)


class StochasticOptimizerTests(unittest.TestCase):
    def test_risk_penalty_prefers_stable_window(self):
        scenarios = np.zeros((2, 96), dtype=float)
        scenarios[0, 2:4] = 15
        scenarios[1, 2:4] = -3
        scenarios[:, 4:6] = 4

        risk_neutral = optimize_one_day_scenarios(
            scenarios,
            block_size=2,
            power_value=1.0,
            discharge_start_min=2,
            discharge_start_max=4,
            risk_lambda=0.0,
        )
        risk_aware = optimize_one_day_scenarios(
            scenarios,
            block_size=2,
            power_value=1.0,
            discharge_start_min=2,
            discharge_start_max=4,
            risk_lambda=0.5,
        )

        self.assertEqual(risk_neutral.discharge_start, 2)
        self.assertEqual(risk_aware.discharge_start, 4)
        self.assertGreater(risk_neutral.profit_std, risk_aware.profit_std)

    def test_detect_scenario_columns_prefers_seed_predictions(self):
        df = pd.DataFrame(
            {
                "times": pd.date_range("2026-01-01", periods=2, freq="15min"),
                "实时价格": [1.0, 2.0],
                "pred_price_seed42": [1.1, 2.1],
                "pred_price_seed2024": [0.9, 1.9],
                "pred_q10": [0.5, 1.5],
                "pred_q50": [1.0, 2.0],
                "pred_q90": [1.5, 2.5],
            }
        )

        self.assertEqual(
            detect_scenario_columns(df),
            ["pred_price_seed42", "pred_price_seed2024"],
        )

    def test_detect_scenario_columns_prefers_residual_scenarios(self):
        df = pd.DataFrame(
            {
                "times": pd.date_range("2026-01-01", periods=2, freq="15min"),
                "price": [1.0, 2.0],
                "pred_price_seed42": [1.1, 2.1],
                "resid_scenario_001": [0.9, 1.9],
                "resid_scenario_000": [1.0, 2.0],
            }
        )

        self.assertEqual(
            detect_scenario_columns(df),
            ["resid_scenario_000", "resid_scenario_001"],
        )

    def test_candidate_pool_deduplicates_and_ranks(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        price = np.zeros(96, dtype=float)
        price[10:18] = -1.0
        price[30:38] = 5.0
        price[40:48] = 3.0
        price_df = pd.DataFrame(
            {
                "times": times,
                "瀹炴椂浠锋牸": price,
                "pred_price_seed42": price,
                "pred_price_seed2024": price,
            }
        )
        reference = pd.DataFrame(
            {
                "times": times,
                "瀹炴椂浠锋牸": price,
                "power": np.zeros(96, dtype=float),
            }
        )
        reference.loc[12:19, "power"] = -1000.0
        reference.loc[40:47, "power"] = 1000.0

        pool = build_candidate_pool(
            price_df,
            reference,
            scenario_sets=[("seed_pair", ["pred_price_seed42", "pred_price_seed2024"])],
            risk_lambdas=[0.0, 0.25],
            max_abs_start_deltas=[20],
            min_delta_score=0.0,
            blocked_dates=set(),
        )

        self.assertEqual(len(pool), 1)
        self.assertEqual(pool.iloc[0]["candidate_charge_start"], 10)
        self.assertEqual(pool.iloc[0]["candidate_discharge_start"], 30)

    def test_apply_single_day_action_only_changes_selected_day(self):
        times = pd.date_range("2026-01-01", periods=192, freq="15min")
        submission = pd.DataFrame(
            {
                "times": times,
                "瀹炴椂浠锋牸": np.zeros(192),
                "power": np.zeros(192),
            }
        )
        selected = pd.Series(
            {
                "date": "2026-01-02",
                "candidate_charge_start": 1,
                "candidate_discharge_start": 20,
            }
        )

        out = apply_single_day_action(submission, selected)
        day1 = out[pd.to_datetime(out["times"]).dt.date.astype(str).eq("2026-01-01")]
        day2 = out[pd.to_datetime(out["times"]).dt.date.astype(str).eq("2026-01-02")]

        self.assertEqual(int((day1["power"] != 0).sum()), 0)
        self.assertEqual(int((day2["power"] < 0).sum()), 8)
        self.assertEqual(int((day2["power"] > 0).sum()), 8)

    def test_default_scenario_sets_include_seed_pairs(self):
        df = pd.DataFrame(
            {
                "pred_price_seed42": [1.0],
                "pred_price_seed2024": [1.0],
                "pred_price_seed2026": [1.0],
            }
        )

        names = [name for name, _ in default_scenario_sets(df)]
        self.assertIn("all_seed", names)
        self.assertIn("seed_pair_pred_price_seed42_pred_price_seed2024", names)

    def test_default_scenario_sets_use_residual_scenarios_first(self):
        df = pd.DataFrame(
            {
                "pred_price_seed42": [1.0],
                "pred_price_seed2024": [1.0],
                "resid_scenario_000": [1.0],
                "resid_scenario_001": [1.0],
            }
        )

        self.assertEqual(
            default_scenario_sets(df),
            [("residual_day_resample", ["resid_scenario_000", "resid_scenario_001"])],
        )

    def test_filter_candidate_pool_requires_conservative_settings(self):
        pool = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "scenario_set": "seed_pair",
                    "risk_lambda": 0.0,
                    "top1_top2_margin": 0.01,
                    "delta_score": 100.0,
                    "expected_delta_profit": 100.0,
                },
                {
                    "date": "2026-01-02",
                    "scenario_set": "all_seed",
                    "risk_lambda": 0.25,
                    "top1_top2_margin": 2.0,
                    "delta_score": 50.0,
                    "expected_delta_profit": 50.0,
                },
            ]
        )

        filtered = filter_candidate_pool(
            pool,
            require_scenario_set="all_seed",
            min_risk_lambda=0.1,
            min_top1_top2_margin=1.0,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["date"], "2026-01-02")

    def test_filter_candidate_pool_can_require_submission_price_delta(self):
        pool = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "scenario_set": "residual_day_resample",
                    "risk_lambda": 0.5,
                    "top1_top2_margin": 1.0,
                    "delta_score": 100.0,
                    "expected_delta_profit": 100.0,
                    "submission_price_delta": -1.0,
                    "multi_price_delta_agree": False,
                },
                {
                    "date": "2026-01-02",
                    "scenario_set": "residual_day_resample",
                    "risk_lambda": 0.5,
                    "top1_top2_margin": 1.0,
                    "delta_score": 50.0,
                    "expected_delta_profit": 50.0,
                    "submission_price_delta": 2.0,
                    "multi_price_delta_agree": True,
                },
            ]
        )

        filtered = filter_candidate_pool(
            pool,
            min_submission_price_delta=0.0,
            require_multi_price_agree=True,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["date"], "2026-01-02")

    def test_seed_delta_diagnostics_filter_all_seed_agreement(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        price_df = pd.DataFrame(
            {
                "times": times,
                "pred_price_seed42": np.arange(96, dtype=float),
                "pred_price_seed2024": np.arange(96, dtype=float),
                "pred_price_seed2026": np.arange(96, dtype=float),
            }
        )
        pool = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "baseline_charge_start": 10,
                    "baseline_discharge_start": 20,
                    "candidate_charge_start": 8,
                    "candidate_discharge_start": 25,
                    "scenario_set": "seed_pair",
                    "risk_lambda": 0.5,
                    "top1_top2_margin": 1.0,
                    "delta_score": 100.0,
                    "expected_delta_profit": 100.0,
                }
            ]
        )

        diagnosed = add_seed_delta_diagnostics(pool, price_df, power_value=1.0)
        filtered = filter_candidate_pool(
            diagnosed,
            min_risk_lambda=0.1,
            min_all_seed_delta=0.1,
            min_all_seed_positive_count=3,
        )

        self.assertEqual(len(filtered), 1)
        self.assertGreater(filtered.iloc[0]["all_seed_delta_min"], 0.0)


if __name__ == "__main__":
    unittest.main()
