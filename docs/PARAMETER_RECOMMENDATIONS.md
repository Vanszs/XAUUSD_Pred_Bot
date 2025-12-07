# Parameter & Configuration Recommendations
Based on "Improved Gold Price Prediction Based on the LSTM-ARIMA Hybrid Model" (CONF-MLA 2025) and your custom 2-year Hourly dataset.

## 1. Dataset Context
*   **Paper**: 11 Years of **Daily** Data (~2800 samples). Less noise, strong long-term trends.
*   **You**: 2 Years of **Hourly** Data (~12,000+ samples). High frequency, more noise, intraday volatility.

## 2. Recommended Hyperparameters

| Parameter | Paper (Inferred/Standard) | **Recommended for YOU (Hourly)** | Reason |
| :--- | :--- | :--- | :--- |
| **Input Data** | Raw Prices ($P_t$) | **Log Returns** ($\ln(P_t/P_{t-1})$) | **CRITICAL**: Fixing the "Gap". Hourly prices drift significantly. Returns are stationary, preventing model collapse on new price levels. |
| **Look Back** | Likely 60-90 days | **120 steps** (5 days) | 60 steps = 2.5 days. 120 steps captures a full trading week (Mon-Fri) pattern. |
| **Epochs** | Likely 50-100 | **50** (with Early Stopping) | Hourly data is larger; 20 epochs is too few for convergence. |
| **Batch Size** | 32 or 64 | **64** | Larger dataset generally benefits from slightly larger batch size for stable gradients. |
| **LSTM Units** | 128 / 64 | **128 / 64** (Keep) | Sufficient complexity. Don't go smaller. |
| **Dropout** | 0.2 | **0.2 - 0.3** | Hourly data is noisier; slightly higher dropout prevents overfitting to noise. |
| **ARIMA Order** | (5,1,0) | **(5,0,0)** (if using Returns) | If inputs are *already* differenced (Returns), ARIMA `d` should be 0. |

## 3. What is Needed? (Action Items)

### A. Implement Stationarity (Log Returns)
The "Prediction Gap" (where prediction line is far below actual) happens because the LSTM trained on prices like \$2000-\$2400 and is now seeing \$2600+. It outputs the average of what it knows (~$2200).
*   **Solution**: Train on **% Change**. If price goes \$2600 -> \$2603, the change is +0.1%. The LSTM knows how to predict "+0.1%", regardless of whether price is \$2000 or \$3000.

### B. Increase Context Window
*   Shift `LOOK_BACK` from `60` (60 hours) to `120` (120 hours / 5 trading days). This allows the model to "see" the weekly cycle.

### C. Update ARIMA for Returns
*   Change ARIMA Order from `(5,1,0)` to `(5,0,0)` because Log Returns are already differenced. Double differencing causes overfitting.

### D. Re-train with more Epochs
*   Increase `EPOCHS` to 50 to allow the model to settle the loss on the large dataset.
