import unittest

import numpy as np
import pandas as pd

from src.breakeven_candidate_scanner import (
    add_residual_scenario_features,
    build_breakeven_candidate_pool,
)


class BreakevenCandidateScannerTests(unittest.TestCase):
    def test_residual_scenario_features_measure_action_delta(self):
        actions = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "baseline_charge_start": 1,
                    "baseline_discharge_start": 10,
                    "candidate_charge_start": 2,
                    "candidate_discharge_start": 10,
                }
            ]
        )
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        scenario = np.zeros(96, dtype=float)
        scenario[1:9] = 3.0
        scenario[2:10] = 1.0
        scenario[10:18] = 5.0
        residual_df = pd.DataFrame(
            {
                "times": times,
                "resid_scenario_000": scenario,
                "resid_scenario_001": scenario,
            }
        )

        out = add_residual_scenario_features(actions, residual_df, power_value=1.0)

        self.assertGreater(out.iloc[0]["residual_delta_mean"], 0.0)
        self.assertEqual(out.iloc[0]["residual_positive_rate"], 1.0)

    def test_build_pool_requires_nonnegative_submission_and_positive_scenarios(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        price = np.zeros(96, dtype=float)
        price[10:18] = 4.0
        reference = pd.DataFrame({"times": times, "实时价格": price, "power": np.zeros(96)})
        reference.loc[1:8, "power"] = -1000.0
        reference.loc[10:17, "power"] = 1000.0

        seed = np.zeros(96, dtype=float)
        seed[1:9] = 3.0
        seed[2:10] = 1.0
        seed[10:18] = 5.0
        seed_df = pd.DataFrame(
            {
                "times": times,
                "pred_price_seed42": seed,
                "pred_price_seed2024": seed,
                "pred_price_seed2026": seed,
            }
        )
        residual_df = pd.DataFrame(
            {
                "times": times,
                "resid_scenario_000": seed,
                "resid_scenario_001": seed,
            }
        )

        pool = build_breakeven_candidate_pool(
            reference,
            seed_df,
            residual_df,
            max_shift=1,
            min_pred_seed_delta=1.0,
            min_residual_p10_delta=1.0,
            min_residual_positive_rate=1.0,
            max_selected_abs_start_delta=1,
        )

        self.assertGreaterEqual(len(pool), 1)
        self.assertEqual(int(pool.iloc[0]["candidate_charge_start"]), 2)
        self.assertEqual(float(pool.iloc[0]["submission_price_delta"]), 0.0)
        self.assertTrue(bool(pool.iloc[0]["multi_price_delta_agree"]))


if __name__ == "__main__":
    unittest.main()
