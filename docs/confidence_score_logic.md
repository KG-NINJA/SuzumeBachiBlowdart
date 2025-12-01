# Confidence Score Calculation

This document outlines the mathematical logic behind the unified `confidence_score` used throughout the SuzumeBachiBlowdart system.

## Formula

The confidence score is calculated in three steps: **Base Score**, **Accuracy Adjustment**, and **Regime Adjustment**.

### 1. Base Score
The base score represents the raw distance from a neutral probability (0.5), normalized to a 0.0-1.0 scale.

$$ Score_{base} = |P_{pred} - 0.5| \times 2 $$

Where:
- $P_{pred}$ is the prediction probability output by the model (0.0 to 1.0).
- $0.5$ represents the neutral threshold (uncertainty).
- $\times 2$ scales the result so that a probability of 1.0 or 0.0 yields a score of 1.0.

### 2. Accuracy Adjustment (Optional)
If the model's historical accuracy is known, the score is adjusted. Higher accuracy boosts confidence; lower accuracy dampens it.

$$ Factor_{acc} = 0.5 + Acc_{model} $$
$$ Score_{adjusted} = Score_{base} \times Factor_{acc} $$

Where:
- $Acc_{model}$ is the model's test accuracy (e.g., 0.60 for 60%).
- A baseline accuracy of 50% ($0.5$) results in a factor of $1.0$ (no change).
- Accuracy > 50% boosts the score (e.g., 60% accuracy -> 1.1x multiplier).

### 3. Final Score
The score is further adjusted by a market regime factor and clipped to the [0.0, 1.0] range.

$$ Score_{final} = \text{clip}(Score_{adjusted} \times F_{regime}, 0.0, 1.0) $$

Where:
- $F_{regime}$ is the market regime factor (default 1.0).
- `clip` ensures the final score never exceeds 1.0 or falls below 0.0.

---

## Logic & Interpretation

The score transforms a raw probability into an actionable "Confidence Level".

- **0.0 (0%)**: The model is completely uncertain (probability 0.5).
- **1.0 (100%)**: The model is mathematically certain (probability 0.0 or 1.0), optionally boosted by high model accuracy.

### Thresholds

| Level | Score Range | Interpretation | Action |
|-------|-------------|----------------|--------|
| **STRONG** | $\ge 0.30$ | High conviction signal. | **EXECUTE** |
| **MEDIUM** | $0.10 \le x < 0.30$ | Moderate signal. | **HOLD / WATCH** |
| **WEAK** | $< 0.10$ | Low conviction, likely noise. | **SKIP** |

---

## Examples

Assuming $F_{regime} = 1.0$ (Neutral Market).

### Case 1: Neutral Probability
- **Input**: $P_{pred} = 0.50$
- **Calculation**: $|0.50 - 0.5| \times 2 = 0.0$
- **Result**: **0.00 (WEAK)**
- **Meaning**: Complete uncertainty.

### Case 2: Slight Lean
- **Input**: $P_{pred} = 0.55$
- **Calculation**: $|0.55 - 0.5| \times 2 = 0.10$
- **Result**: **0.10 (MEDIUM)**
- **Meaning**: Just enough signal to consider, but risky.

### Case 3: Moderate Signal
- **Input**: $P_{pred} = 0.65$
- **Calculation**: $|0.65 - 0.5| \times 2 = 0.30$
- **Result**: **0.30 (STRONG)**
- **Meaning**: The threshold for a "Strong" signal. Corresponds to 65% probability.

### Case 4: High Confidence
- **Input**: $P_{pred} = 0.80$
- **Calculation**: $|0.80 - 0.5| \times 2 = 0.60$
- **Result**: **0.60 (STRONG)**
- **Meaning**: Very strong signal.

### Case 5: With Model Accuracy Boost
- **Input**: $P_{pred} = 0.60$, $Acc_{model} = 0.60$ (60% accuracy)
- **Base Score**: $|0.60 - 0.5| \times 2 = 0.20$
- **Adjustment**: $0.20 \times (0.5 + 0.60) = 0.20 \times 1.1 = 0.22$
- **Result**: **0.22 (MEDIUM)**
- **Meaning**: A 60% probability is usually "Medium" (0.20), but the model's good accuracy boosts it slightly to 0.22.
