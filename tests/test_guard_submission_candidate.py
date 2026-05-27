import unittest

import pandas as pd

from src.guard_submission_candidate import _guard_manifest_policy


class GuardSubmissionCandidateTests(unittest.TestCase):
    def test_manifest_policy_rejects_negative_submission_price_delta(self):
        manifest_row = pd.Series(
            {
                "candidate_sha256": "ABC",
                "date": "2026-02-01",
                "blocked": False,
                "changed_days": 1,
                "baseline_charge_start": 10,
                "baseline_discharge_start": 20,
                "candidate_charge_start": 9,
                "candidate_discharge_start": 22,
                "pred_window_score": 100.0,
                "score_std": 1.0,
                "top1_top2_margin": 5.0,
                "reason": "unit test",
                "submission_price_delta": -0.1,
                "multi_price_delta_agree": True,
            }
        )
        summary = {"changed_days": 1}
        changed_actions = [
            {
                "date": "2026-02-01",
                "candidate_action": "charge=9-16;discharge=22-29",
            }
        ]

        errors, info = _guard_manifest_policy(manifest_row, summary, changed_actions, "ABC")

        self.assertIn("manifest submission_price_delta is negative: -0.1", errors)
        self.assertEqual(info["submission_price_delta"], -0.1)

    def test_manifest_policy_rejects_multi_price_disagreement(self):
        manifest_row = pd.Series(
            {
                "candidate_sha256": "ABC",
                "date": "2026-02-01",
                "blocked": False,
                "changed_days": 1,
                "baseline_charge_start": 10,
                "baseline_discharge_start": 20,
                "candidate_charge_start": 9,
                "candidate_discharge_start": 22,
                "pred_window_score": 100.0,
                "score_std": 1.0,
                "top1_top2_margin": 5.0,
                "reason": "unit test",
                "submission_price_delta": 1.0,
                "multi_price_delta_agree": False,
            }
        )
        summary = {"changed_days": 1}
        changed_actions = [
            {
                "date": "2026-02-01",
                "candidate_action": "charge=9-16;discharge=22-29",
            }
        ]

        errors, _ = _guard_manifest_policy(manifest_row, summary, changed_actions, "ABC")

        self.assertIn("manifest multi_price_delta_agree is false", errors)

    def test_portfolio_manifest_allows_acknowledged_multi_day_candidate(self):
        manifest_row = pd.Series(
            {
                "candidate_csv": "outputs/output_portfolio.csv",
                "candidate_sha256": "ABC",
                "manifest_stage": "portfolio_high_upside",
                "blocked": False,
                "portfolio_acknowledged_high_risk": True,
                "changed_days": 2,
                "selected_dates": "2026-02-01,2026-02-03",
                "pred_window_score": 1000.0,
                "expected_delta_profit": 1200.0,
                "score_std": 10.0,
                "top1_top2_margin": 1.0,
                "reason": "unit test portfolio",
                "submission_price_delta": -5.0,
            }
        )
        summary = {"changed_days": 2}
        changed_actions = [
            {
                "date": "2026-02-01",
                "candidate_action": "charge=9-16;discharge=22-29",
            },
            {
                "date": "2026-02-03",
                "candidate_action": "charge=10-17;discharge=23-30",
            },
        ]

        errors, info = _guard_manifest_policy(manifest_row, summary, changed_actions, "ABC")

        self.assertEqual(errors, [])
        self.assertEqual(info["stage"], "portfolio_high_upside")
        self.assertEqual(info["submission_price_delta"], -5.0)

    def test_portfolio_manifest_requires_high_risk_acknowledgement(self):
        manifest_row = pd.Series(
            {
                "candidate_csv": "outputs/output_portfolio.csv",
                "candidate_sha256": "ABC",
                "manifest_stage": "portfolio_high_upside",
                "blocked": False,
                "portfolio_acknowledged_high_risk": False,
                "changed_days": 1,
                "selected_dates": "2026-02-01",
                "pred_window_score": 1000.0,
                "expected_delta_profit": 1200.0,
                "score_std": 10.0,
                "top1_top2_margin": 1.0,
                "reason": "unit test portfolio",
            }
        )
        summary = {"changed_days": 1}
        changed_actions = [
            {
                "date": "2026-02-01",
                "candidate_action": "charge=9-16;discharge=22-29",
            }
        ]

        errors, _ = _guard_manifest_policy(manifest_row, summary, changed_actions, "ABC")

        self.assertIn("portfolio manifest must set portfolio_acknowledged_high_risk=true", errors)


if __name__ == "__main__":
    unittest.main()
