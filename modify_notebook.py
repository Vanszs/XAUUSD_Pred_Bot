import json
import os

NOTEBOOK_PATH = '/home/vanszs/Code/XAUUSD_Pred_Bot/analysis_bigru.ipynb'

def modify_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    modified_count = 0

    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        
        source_lines = cell.get('source', [])
        source_text = "".join(source_lines)

        # 1. Update Configuration Cell
        if "# --- CONFIGURATION" in source_text and "BACKTEST_MODE" not in source_text:
            print("Found Configuration Cell. Updating...")
            
            # Find the line with PERIOD and insert BACKTEST configs after it
            new_source = []
            for line in source_lines:
                new_source.append(line)
                if "PERIOD =" in line:
                    new_source.append("\n")
                    new_source.append("# Backtest Mode (True = Train on history, Test on last 7 days)\n")
                    new_source.append("BACKTEST_MODE = True\n")
                    new_source.append("BACKTEST_DAYS = 7\n")
            
            cell['source'] = new_source
            modified_count += 1

        # 2. Update Data Split Cell
        elif "# Prepare data with paper-recommended look_back" in source_text:
            print("Found Data Split Cell. Updating logic...")
            
            # We will replace the standard split logic with the conditional logic
            # Existing specific lines to replace:
            # train_size = int(len(X) * 0.8)
            # X_train, X_test = X[:train_size], X[train_size:]
            # y_train, y_test = y[:train_size], y[train_size:]
            
            new_code = [
                "# Prepare data with paper-recommended look_back\n",
                "X, y, scaled_data = loader.prepare_data_for_lstm(df, look_back=LOOK_BACK)\n",
                "\n",
                "if BACKTEST_MODE:\n",
                "    # Calculate cutoff for last N days\n",
                "    last_idx = df.index[-1]\n",
                "    split_date = last_idx - pd.Timedelta(days=BACKTEST_DAYS)\n",
                "    \n",
                "    print(f\"⚠️ BACKTEST MODE ON\")\n",
                "    print(f\"  Training End / Test Start: {split_date}\")\n",
                "    \n",
                "    # Count how many candles are in the test period\n",
                "    test_rows = df[df.index > split_date]\n",
                "    n_test = len(test_rows)\n",
                "    \n",
                "    # Limit n_test to prevent index errors\n",
                "    n_test = min(n_test, int(len(y) * 0.5))\n",
                "    \n",
                "    train_size = len(y) - n_test\n",
                "    print(f\"  Test Samples (Last {BACKTEST_DAYS} days): {n_test}\")\n",
                "else:\n",
                "    # Standard 80/20 Split\n",
                "    train_size = int(len(X) * 0.8)\n",
                "\n",
                "X_train, X_test = X[:train_size], X[train_size:]\n",
                "y_train, y_test = y[:train_size], y[train_size:]\n",
                "\n",
                "print(f'Train samples: {X_train.shape[0]}')\n",
                "print(f'Test samples: {X_test.shape[0]}')\n",
                "print(f'Input shape: {X_train.shape}')\n"
            ]
            
            cell['source'] = new_code
            modified_count += 1

    if modified_count > 0:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully modified {modified_count} cells.")
    else:
        print("No matching cells found or already modified.")

if __name__ == "__main__":
    modify_notebook()
