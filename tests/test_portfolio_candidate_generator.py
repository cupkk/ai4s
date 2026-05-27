import unittest

import numpy as np
import pandas as pd

from src.portfolio_candidate_generator import (
    add_calendar_replay_diagnostics,
    apply_portfolio_actions,
    build_calendar_action_pool,
    build_calendar_oracle_windows,
    expand_top_counts_for_target,
)


class PortfolioCandidateGeneratorTests(unittest.TestCase):
    def test_calendar_oracle_windows_extracts_best_charge_discharge_pair(self):
        times = pd.date_range("2025-01-01", periods=96, freq="15min")
        prices = np.zeros(96, dtype=float)
        prices[10:18] = -2.0
        prices[40:48] = 5.0
        label = pd.DataFrame({"times": times, "A": prices})

        windows = build_calendar_oracle_windows(label)

        self.assertEqual(len(windows), 1)
        self.assertEqual(int(windows.iloc[0]["hist_charge_start"]), 10)
        self.assertEqual(int(windows.iloc[0]["hist_discharge_start"]), 40)

    def test_calendar_action_pool_can_cap_large_historical_shift(self):
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        reference = pd.DataFrame({"times": times, "实时价格": np.zeros(96), "power": np.zeros(96)})
        reference.loc[20:27, "power"] = -1000.0
        reference.loc[60:67, "power"] = 1000.0
        calendar = pd.DataFrame(
            [
                {
                    "month_day": "01-01",
                    "hist_date": "2025-01-01",
                    "hist_charge_start": 10,
                    "hist_discharge_start": 80,
                    "hist_oracle_profit": 1.0,
                }
            ]
        )

        pool = build_calendar_action_pool(reference, calendar, cap_shift=4, blocked_dates=set())

        self.assertEqual(int(pool.iloc[0]["candidate_charge_start"]), 16)
        self.assertEqual(int(pool.iloc[0]["candidate_discharge_start"]), 64)
        self.assertEqual(int(pool.iloc[0]["max_abs_start_delta"]), 4)

    def test_apply_portfolio_actions_changes_only_selected_dates(self):
        times = pd.date_range("2026-01-01", periods=192, freq="15min")
        reference = pd.DataFrame({"times": times, "实时价格": np.zeros(192), "power": np.zeros(192)})
        actions = pd.DataFrame(
            [
                {
                    "date": "2026-01-02",
                    "candidate_charge_start": 3,
                    "candidate_discharge_start": 20,
                }
            ]
        )

        out = apply_portfolio_actions(reference, actions)
        out["times"] = pd.to_datetime(out["times"])
        day1 = out.loc[out["times"].dt.date.astype(str).eq("2026-01-01")]
        day2 = out.loc[out["times"].dt.date.astype(str).eq("2026-01-02")]

        self.assertEqual(int((day1["power"] != 0).sum()), 0)
        self.assertEqual(int((day2["power"] < 0).sum()), 8)
        self.assertEqual(int((day2["power"] > 0).sum()), 8)

    def test_calendar_replay_diagnostics_scores_candidate_against_baseline(self):
        times = pd.date_range("2025-01-01", periods=96, freq="15min")
        prices = np.zeros(96, dtype=float)
        prices[5:13] = -1.0
        prices[40:48] = 2.0
        prices[20:28] = -0.5
        prices[60:68] = 1.0
        label = pd.DataFrame({"times": times, "A": prices})
        actions = pd.DataFrame(
            [
                {
                    "month_day": "01-01",
                    "baseline_charge_start": 20,
                    "baseline_discharge_start": 60,
                    "candidate_charge_start": 5,
                    "candidate_discharge_start": 40,
                }
            ]
        )

        out = add_calendar_replay_diagnostics(actions, label)

        self.assertAlmostEqual(float(out.iloc[0]["hist_baseline_profit"]), 12000.0)
        self.assertAlmostEqual(float(out.iloc[0]["hist_candidate_profit"]), 24000.0)
        self.assertAlmostEqual(float(out.iloc[0]["hist_delta_profit"]), 12000.0)

    def test_expand_top_counts_for_target_adds_minimum_crossing_count(self):
        reference_times = pd.date_range("2026-01-01", periods=96 * 2, freq="15min")
        reference = pd.DataFrame(
            {"times": reference_times, "price": np.zeros(96 * 2), "power": np.zeros(96 * 2)}
        )
        pool = pd.DataFrame(
            {
                "portfolio_rank_score": [3.0, 2.0, 1.0],
                "hist_delta_profit": [100.0, 100.0, 100.0],
            }
        )

        counts = expand_top_counts_for_target(
            pool,
            top_counts=[1],
            reference_submission=reference,
            reference_score=10.0,
            target_score=110.0,
        )

        self.assertEqual(counts, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
