import unittest

import numpy as np
import pandas as pd

from src.residual_scenario_generator import (
    DEFAULT_BASE_COL,
    build_daily_residual_library,
    generate_residual_scenarios,
    residual_library_diagnostics,
    scenario_diagnostics,
)


class ResidualScenarioGeneratorTests(unittest.TestCase):
    def _validation_frame(self) -> pd.DataFrame:
        times = pd.date_range("2025-01-01", periods=192, freq="15min")
        base = np.full(192, 10.0)
        residual = np.r_[np.arange(96, dtype=float), 100.0 + np.arange(96, dtype=float)]
        return pd.DataFrame(
            {
                "times": times,
                "A": base + residual,
                "pred_price_seed42": base - 1.0,
                "pred_price_seed2024": base + 1.0,
            }
        )

    def _test_frame(self) -> pd.DataFrame:
        times = pd.date_range("2026-01-01", periods=96, freq="15min")
        return pd.DataFrame(
            {
                "times": times,
                "price": np.linspace(1.0, 2.0, 96),
                "pred_price_seed42": np.full(96, 20.0),
                "pred_price_seed2024": np.full(96, 22.0),
            }
        )

    def test_builds_complete_daily_residual_vectors(self):
        library = build_daily_residual_library(self._validation_frame())

        self.assertEqual(library.residuals.shape, (2, 96))
        self.assertEqual(library.source_dates, ["2025-01-01", "2025-01-02"])
        np.testing.assert_allclose(library.residuals[0], np.arange(96, dtype=float))

    def test_generated_scenarios_reuse_whole_day_residual_vectors(self):
        library = build_daily_residual_library(self._validation_frame())
        scenarios, source_map = generate_residual_scenarios(
            self._test_frame(),
            library,
            n_scenarios=4,
            random_seed=7,
            price_col="price",
        )

        day = scenarios.sort_values("times").reset_index(drop=True)
        base = day[DEFAULT_BASE_COL].to_numpy(dtype=float)
        for _, row in source_map.iterrows():
            observed_residual = day[row["scenario_col"]].to_numpy(dtype=float) - base
            expected_residual = library.residuals[int(row["source_index"])]
            np.testing.assert_allclose(observed_residual, expected_residual)

    def test_generation_is_reproducible_with_seed(self):
        library = build_daily_residual_library(self._validation_frame())
        first, first_map = generate_residual_scenarios(
            self._test_frame(),
            library,
            n_scenarios=3,
            random_seed=2026,
            price_col="price",
        )
        second, second_map = generate_residual_scenarios(
            self._test_frame(),
            library,
            n_scenarios=3,
            random_seed=2026,
            price_col="price",
        )

        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first_map, second_map)

    def test_diagnostics_report_autocorrelation_and_source_coverage(self):
        library = build_daily_residual_library(self._validation_frame())
        scenarios, source_map = generate_residual_scenarios(
            self._test_frame(),
            library,
            n_scenarios=4,
            random_seed=7,
            price_col="price",
        )

        scenario_diag = scenario_diagnostics(scenarios, source_map)
        library_diag = residual_library_diagnostics(library)

        self.assertEqual(len(scenario_diag), 1)
        self.assertEqual(int(scenario_diag.iloc[0]["n_scenarios"]), 4)
        self.assertGreaterEqual(int(scenario_diag.iloc[0]["unique_source_days"]), 1)
        self.assertEqual(int(library_diag.iloc[0]["residual_days"]), 2)


if __name__ == "__main__":
    unittest.main()
