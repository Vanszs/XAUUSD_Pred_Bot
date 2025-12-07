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
| Data Loader | `src/data_loader.py` | Platform auto-detect (MT5/YFinance), OHLCV multivariate |
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

## Data Sources (Auto-Detect)

Data source is automatically selected based on platform:

| Platform | Data Source | Symbol |
|----------|-------------|--------|
| **Windows** | MetaTrader 5 | `XAUUSD` |
| **Linux/WSL** | Yahoo Finance | `GC=F` (Gold Futures) |

```python
# src/data_loader.py
loader.fetch_data(source='auto')  # Auto-detect platform
```

### Manual Override
```python
loader.fetch_data(source='mt5')       # Force MetaTrader5
loader.fetch_data(source='yfinance')  # Force Yahoo Finance
```

---

## GPU Support

### Current Setup (WSL2 Ubuntu)
- **GPU**: NVIDIA RTX 3050 (4GB VRAM)
- **TensorFlow**: 2.20.0 with CUDA
- **Status**: ✅ GPU Enabled

### Installation
```bash
# In WSL2 Ubuntu
pip install tensorflow[and-cuda]
```

### Verify GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Model Performance (Baseline)

With 5y data, 1h timeframe, 10 epochs:
- **R²**: 0.9898 (excellent)
- **RMSE**: 0.0136 (normalized scale)
- **MSE**: 0.00018

---

## Known Issues

1. **Future prediction too smooth**: Recursive LSTM accumulates errors
2. **MT5 on Linux**: Not supported - use Yahoo Finance fallback
3. **IPython kernel cache**: After code changes, restart kernel

---

## File Structure

```
XAUUSD_Pred_Bot/
├── analysis.ipynb     # Interactive notebook (7 sections)
├── main.py            # CLI interface
├── requirements.txt   # Dependencies
├── .gitignore         # Excludes venv, data, plots
├── .python-version    # pyenv local (3.11.9)
├── src/
│   ├── __init__.py
│   ├── data_loader.py # Platform auto-detect
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

### Activate Environment
```bash
# Linux/WSL
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### CLI
```bash
python main.py --mode train --interval 1h --epochs 50 --days 5
```

### Notebook
```bash
jupyter notebook analysis.ipynb
# Select kernel: "Python (XAUUSD Venv)"
```

---

## Dependencies

- tensorflow 2.20.0 (with CUDA on WSL2)
- numpy, pandas
- scikit-learn (MinMaxScaler)
- statsmodels (ARIMA)
- matplotlib
- MetaTrader5 (Windows only)
- yfinance
- ipykernel, notebook

---

## Reference

Based on: "Improved Gold Price Prediction Based on the LSTM-ARIMA Hybrid Model" (CONF-MLA 2025)
- PDF: `docs/9a35a69227a649c6a1124bba1ea8d6fb.marked_uJoACAb.pdf`
