"""Generate a markdown summary of recent retraining accuracy results.

The script aggregates ticker-level accuracies from ``retraining_results.csv`` and,
optionally, a historical average from ``accuracy_analysis/analysis_results.json`` to
produce a concise performance snapshot. It is intentionally lightweight and avoids any
network or training side effects so it can be run safely in CI.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Sequence


@dataclass
class TickerResult:
    """A single ticker's retraining outcome."""

    ticker: str
    status: str
    regime: str
    accuracy: float


def _validate_required_columns(fieldnames: Sequence[str] | None, required: set[str]) -> None:
    """Ensure the retraining CSV exposes the columns the summary needs."""

    if not fieldnames:
        raise ValueError("retraining_results.csv is empty or missing headers")

    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"retraining_results.csv is missing required columns: {sorted(missing)}"
        )


def load_retraining_results(path: Path) -> List[TickerResult]:
    """Load retraining outputs and validate the expected schema.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if the
    schema or data is malformed.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Retraining results not found at {path}. Generate retraining_results.csv first."
        )

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        _validate_required_columns(reader.fieldnames, {"ticker", "status", "regime", "accuracy"})

        results: List[TickerResult] = []
        for row in reader:
            try:
                accuracy = float(row["accuracy"])
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid accuracy value for row: {row}") from exc

            results.append(
                TickerResult(
                    ticker=row["ticker"],
                    status=row["status"],
                    regime=row["regime"],
                    accuracy=accuracy,
                )
            )

    if not results:
        raise ValueError("No retraining rows found in retraining_results.csv")

    return results


def summarize_regimes(results: Iterable[TickerResult]):
    """Aggregate accuracy statistics per market regime."""

    regimes = {}
    for row in results:
        regimes.setdefault(row.regime, []).append(row.accuracy)
    return {
        regime: {
            "count": len(values),
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
        }
        for regime, values in regimes.items()
    }


def summarize_statuses(results: Iterable[TickerResult]):
    """Count retraining statuses for quick health checks."""

    counts: dict[str, int] = {}
    for row in results:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def load_historical_average(path: Path) -> float | None:
    """Return the stored average accuracy if available."""

    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)

    stats = data.get("statistics")
    if not stats or "average_accuracy" not in stats:
        return None
    return float(stats["average_accuracy"])


def _format_regime_table(regime_stats: dict[str, dict[str, float]]) -> list[str]:
    if not regime_stats:
        return ["No regime information available."]

    lines = ["Regime | Tickers | Mean | Min | Max", "--- | --- | --- | --- | ---"]
    for regime, stats in sorted(regime_stats.items()):
        lines.append(
            f"{regime} | {stats['count']} | {stats['mean']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f}"
        )
    return lines


def build_report(results: List[TickerResult], historical_avg: float | None) -> str:
    """Render the performance summary to markdown."""

    accuracies = [r.accuracy for r in results]
    regime_stats = summarize_regimes(results)
    status_counts = summarize_statuses(results)
    top = sorted(results, key=lambda r: r.accuracy, reverse=True)[:3]
    bottom = sorted(results, key=lambda r: r.accuracy)[:3]

    lines = [
        "# Stock Prediction Performance",
        "",
        f"Generated from `retraining_results.csv` containing {len(results)} tickers.",
        "",
        "## Current retraining snapshot",
        "",
        f"- Mean accuracy: {mean(accuracies):.4f}",
        f"- Median accuracy: {median(accuracies):.4f}",
        f"- Status breakdown: {status_counts}",
        f"- Best ticker: {top[0].ticker} ({top[0].accuracy:.4f})",
        f"- Lowest ticker: {bottom[0].ticker} ({bottom[0].accuracy:.4f})",
        "",
        "### Regime breakdown",
        *_format_regime_table(regime_stats),
        "",
        "### Top performers",
    ]

    for row in top:
        lines.append(f"- {row.ticker}: {row.accuracy:.4f} (status={row.status})")

    lines.append("\n### Bottom performers")
    for row in bottom:
        lines.append(f"- {row.ticker}: {row.accuracy:.4f} (status={row.status})")

    if historical_avg is not None:
        lines.extend(
            [
                "",
                "## Historical context",
                f"- Previous average accuracy (from `accuracy_analysis/analysis_results.json`): {historical_avg:.4f}",
                f"- Change vs current mean: {mean(accuracies) - historical_avg:+.4f}",
            ]
        )

    lines.append("\n## Notes")
    lines.append("- All figures are derived from existing offline evaluation artifacts; no live trading signals were generated.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retraining-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "retraining_results.csv",
        help="Path to retraining_results.csv produced by the training pipeline.",
    )
    parser.add_argument(
        "--historical-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "accuracy_analysis"
        / "analysis_results.json",
        help="Optional historical accuracy summary to compare against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "stock_prediction_performance.md",
        help="Where to write the generated markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = load_retraining_results(args.retraining_csv)
    historical_avg = load_historical_average(args.historical_json)

    report_text = build_report(results, historical_avg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text)


if __name__ == "__main__":
    main()
