import unittest

import pandas as pd

from pipeline_safety import (
    apply_confidence_decisions,
    classify_run_status,
    normalize_ohlcv_columns,
    normalize_prediction_probabilities,
    should_fail_run,
)


class NormalizeOhlcvColumnsTests(unittest.TestCase):
    def test_normalizes_flat_columns(self):
        frame = pd.DataFrame(
            [[1, 2, 0, 1.5, 100]],
            columns=["open", "HIGH", "low", "Close", "volume"],
        )

        result = normalize_ohlcv_columns(frame, ticker="TEST")

        self.assertEqual(
            list(result.columns), ["Open", "High", "Low", "Close", "Volume"]
        )
        self.assertEqual(list(frame.columns), ["open", "HIGH", "low", "Close", "volume"])

    def test_normalizes_yfinance_price_first_multiindex(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Open", "AAPL"),
                ("High", "AAPL"),
                ("Low", "AAPL"),
                ("Close", "AAPL"),
                ("Volume", "AAPL"),
            ]
        )
        frame = pd.DataFrame([[1, 2, 0, 1.5, 100]], columns=columns)

        result = normalize_ohlcv_columns(frame, ticker="AAPL")

        self.assertEqual(
            list(result.columns), ["Open", "High", "Low", "Close", "Volume"]
        )

    def test_normalizes_yfinance_ticker_first_multiindex(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("AAPL", "Open"),
                ("AAPL", "High"),
                ("AAPL", "Low"),
                ("AAPL", "Close"),
                ("AAPL", "Volume"),
            ]
        )
        frame = pd.DataFrame([[1, 2, 0, 1.5, 100]], columns=columns)

        result = normalize_ohlcv_columns(frame, ticker="AAPL")

        self.assertEqual(
            list(result.columns), ["Open", "High", "Low", "Close", "Volume"]
        )

    def test_rejects_ambiguous_multi_ticker_columns(self):
        columns = pd.MultiIndex.from_tuples(
            [
                (field, ticker)
                for field in ("Open", "High", "Low", "Close", "Volume")
                for ticker in ("AAPL", "MSFT")
            ]
        )
        frame = pd.DataFrame([[1] * len(columns)], columns=columns)

        with self.assertRaisesRegex(ValueError, "ambiguous OHLCV"):
            normalize_ohlcv_columns(frame)

    def test_rejects_missing_required_columns(self):
        frame = pd.DataFrame([[1, 2]], columns=["Close", "Volume"])

        with self.assertRaisesRegex(ValueError, "missing required OHLCV"):
            normalize_ohlcv_columns(frame, ticker="TEST")


class PredictionProbabilityTests(unittest.TestCase):
    def test_v2_uses_raw_ensemble_probability_and_fixes_bearish_fields(self):
        result = normalize_prediction_probabilities(
            {
                "ticker": "TEST",
                "ensemble_pred": 0.20,
                "prob_up": 0.80,
                "prob_down": 0.20,
                "confidence": 0.60,
            }
        )

        self.assertAlmostEqual(result["prob_up"], 0.20)
        self.assertAlmostEqual(result["prob_down"], 0.80)
        self.assertAlmostEqual(result["confidence_score"], 0.60)
        self.assertEqual(result["probability_source"], "ensemble_pred")

    def test_low_v2_confidence_is_not_double_transformed(self):
        decisions = apply_confidence_decisions(
            [
                {
                    "ticker": "TEST",
                    "ensemble_pred": 0.60,
                    "confidence": 0.20,
                    "direction": "Bullish",
                }
            ],
            min_confidence=0.30,
        )

        self.assertAlmostEqual(decisions[0]["confidence_score"], 0.20)
        self.assertEqual(decisions[0]["action"], "SKIP")

    def test_strong_bearish_probability_is_executable(self):
        decisions = apply_confidence_decisions(
            [{"ticker": "TEST", "prob_up": 0.20, "direction": "Bearish"}],
            min_confidence=0.30,
        )

        self.assertAlmostEqual(decisions[0]["confidence_score"], 0.60)
        self.assertEqual(decisions[0]["action"], "EXECUTE")


class RunStatusTests(unittest.TestCase):
    def test_total_failure_is_fatal(self):
        status = classify_run_status(
            attempted_symbols=5,
            successful_inferences=0,
            executable_signals=0,
            failed_symbols=5,
        )
        self.assertEqual(status, "FAILED")
        self.assertTrue(should_fail_run(status))

    def test_successful_abstention_is_not_failure(self):
        status = classify_run_status(
            attempted_symbols=5,
            successful_inferences=5,
            executable_signals=0,
            failed_symbols=0,
        )
        self.assertEqual(status, "NO_SIGNAL")
        self.assertFalse(should_fail_run(status))

    def test_partial_failure_without_signal_is_visible(self):
        status = classify_run_status(
            attempted_symbols=5,
            successful_inferences=3,
            executable_signals=0,
            failed_symbols=2,
        )
        self.assertEqual(status, "PARTIAL_NO_SIGNAL")
        self.assertFalse(should_fail_run(status))

    def test_predictions_with_partial_failure_are_visible(self):
        status = classify_run_status(
            attempted_symbols=5,
            successful_inferences=4,
            executable_signals=2,
            failed_symbols=1,
        )
        self.assertEqual(status, "PARTIAL_SUCCESS")

    def test_full_prediction_run(self):
        status = classify_run_status(
            attempted_symbols=5,
            successful_inferences=5,
            executable_signals=2,
            failed_symbols=0,
        )
        self.assertEqual(status, "PREDICTED")


if __name__ == "__main__":
    unittest.main()
