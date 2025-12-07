# XAUUSD Prediction Bot - Knowledge Base

## Project Overview
A Hybrid LSTM-ARIMA model for predicting XAU/USD (Gold) prices. Based on academic research showing that combining LSTM (for non-linear patterns) with ARIMA (for linear residuals) outperforms standalone models.

---

## Architecture

### Hybrid Model Flow
```
Historical Data → LSTM → Primary Prediction
                    ↓
              Residuals = Actual - LSTM_Pred
                    ↓
                 ARIMA → Residual Forecast
                    ↓
         Final = LSTM_Pred + ARIMA_Residual
```

### Current Implementation (v2 Optimized)

| Component | File | Description |
|-----------|------|-------------|
| Data Loader | `src/data_loader.py` | MT5/YFinance data, OHLCV multivariate support |
| LSTM Model | `src/lstm_model.py` | 128→64 units, BatchNorm, Early Stopping |
| ARIMA Model | `src/arima_model.py` | Order (5,1,0) for residual modeling |
| Hybrid Model | `src/hybrid_model.py` | Orchestrates LSTM + ARIMA |
| Utils | `src/utils.py` | Metrics, plotting, time conversion |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 50 | Training iterations |
| `LOOK_BACK` | 90 | Sequence length for LSTM |
| `LSTM_UNITS` | 128, 64 | Units per layer |
| `DROPOUT` | 0.15 | Regularization |
| `LEARNING_RATE` | 0.0005 | Adam optimizer |
| `ARIMA_ORDER` | (5,1,0) | (p, d, q) |

---

## Data Sources

### MetaTrader 5 (Primary)
- Requires MT5 desktop app running
- Symbol: `XAUUSD`
- Timeframes: `1h`, `30m`, `15m`, `4h`, `1d`

### Yahoo Finance (Fallback)
- Symbol: `GC=F` (Gold Futures)
- Auto-fallback if MT5 fails

---

## GPU Support (Windows Issue)

**Problem**: TensorFlow > 2.10 does NOT support native GPU on Windows.

**Solutions**:

| Option | Method | Pros/Cons |
|--------|--------|-----------|
| **WSL2** | `pip install tensorflow[and-cuda]` in Ubuntu | Full CUDA support, recommended |
| **DirectML** | `tensorflow-directml-plugin` | Uses DirectX 12, limited compatibility |
| **CPU** | Default install | Slow but works |

**Current Status**: Running on CPU (TensorFlow 2.20.0)

---

## Model Performance (Baseline)

With 5y data, 1h timeframe, 10 epochs:
- **R²**: 0.9898 (excellent)
- **RMSE**: 0.0136 (normalized scale)
- **MSE**: 0.00018

---

## Known Issues

1. **Future prediction too smooth**: Recursive LSTM accumulates errors, causing unrealistic declining curves
2. **Windows GPU**: Requires WSL2 for proper CUDA support
3. **IPython kernel cache**: After code changes, must restart kernel

---

## File Structure

```
XAUUSD_Pred_Bot/
├── analysis.ipynb     # Interactive notebook
├── main.py            # CLI interface
├── requirements.txt   # Dependencies
├── .gitignore         # Excludes venv, data, plots
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── lstm_model.py
│   ├── arima_model.py
│   ├── hybrid_model.py
│   └── utils.py
├── data/              # CSV cache (gitignored)
├── plots/             # Generated charts (gitignored)
├── docs/              # Documentation
└── venv/              # Virtual environment (gitignored)
```

---

## Usage

### CLI
```powershell
.\venv\Scripts\activate
python main.py --mode train --interval 1h --epochs 50 --days 5
```

### Notebook
```powershell
.\venv\Scripts\activate
jupyter notebook analysis.ipynb
# Select kernel: "Python (XAUUSD Venv)"
```

---

## Dependencies

- tensorflow (2.20.0 CPU / needs WSL2 for GPU)
- numpy, pandas
- scikit-learn (MinMaxScaler)
- statsmodels (ARIMA)
- matplotlib
- MetaTrader5
- yfinance
- ipykernel, notebook

---

## Reference

Based on: "Improved Gold Price Prediction Based on the LSTM-ARIMA Hybrid Model" (CONF-MLA 2025)
- PDF located in: `docs/9a35a69227a649c6a1124bba1ea8d6fb.marked_uJoACAb.pdf`
