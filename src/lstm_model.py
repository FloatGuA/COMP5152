import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
from src.model_cache import is_valid, save_lstm, load_lstm
from src.seed import set_seed, SEED

WINDOW   = 60
EPOCHS   = 100
BATCH    = 32
PATIENCE = 10
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path("results")
MODEL_NAME  = "LSTM"


class LSTMNet(nn.Module):
    def __init__(self, hidden=50):
        super().__init__()
        self.lstm1 = nn.LSTM(1, hidden, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc    = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out)


def _make_sequences(scaled: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _to_tensor(arr):
    return torch.tensor(arr).to(DEVICE)


def predict(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    symbol    = train.name or "unknown"
    train_end = str(train.index[-1].date())
    val_end   = str(val.index[-1].date())

    scaler  = StandardScaler()
    train_s = scaler.fit_transform(train.values.reshape(-1, 1))
    val_s   = scaler.transform(val.values.reshape(-1, 1))
    test_s  = scaler.transform(test.values.reshape(-1, 1))

    set_seed(SEED)
    model = LSTMNet().to(DEVICE)

    if is_valid(symbol, MODEL_NAME, train_end, val_end):
        tqdm.write(f"    [cache hit] LSTM {symbol} — skipping training")
        state_dict, scaler = load_lstm(symbol)
        model.load_state_dict(state_dict)
        # re-transform with loaded scaler
        train_s = scaler.transform(train.values.reshape(-1, 1))
        val_s   = scaler.transform(val.values.reshape(-1, 1))
        test_s  = scaler.transform(test.values.reshape(-1, 1))
    else:
        tv_s = np.concatenate([train_s, val_s])
        X_train, y_train = _make_sequences(train_s, WINDOW)
        X_tv, y_tv = _make_sequences(tv_s, WINDOW)
        X_val = X_tv[-len(val):]
        y_val = y_tv[-len(val):]

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn   = nn.MSELoss()

        best_val, patience_count, best_state = float("inf"), 0, None
        train_losses, val_losses = [], []

        tqdm.write(f"    Device: {DEVICE}  |  train_seq={len(X_train)}  val_seq={len(X_val)}")

        epoch_bar = tqdm(range(EPOCHS), desc=f"    LSTM [{symbol}]", ncols=80,
                         leave=False, unit="ep")
        epoch_start = time.time()

        for epoch in epoch_bar:
            ep_t0 = time.time()
            model.train()
            perm = np.random.permutation(len(X_train))
            batch_loss, n_batches = 0.0, 0
            for start in range(0, len(X_train), BATCH):
                idx = perm[start:start + BATCH]
                xb  = _to_tensor(X_train[idx])
                yb  = _to_tensor(y_train[idx])
                optimizer.zero_grad()
                loss = loss_fn(model(xb).squeeze(), yb.squeeze())
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                n_batches  += 1

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    model(_to_tensor(X_val)).squeeze(),
                    _to_tensor(y_val).squeeze()
                ).item()

            train_loss = batch_loss / max(n_batches, 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            ep_sec = time.time() - ep_t0
            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                ep_s=f"{ep_sec:.2f}s",
                patience=patience_count,
            )

            if val_loss < best_val:
                best_val       = val_loss
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    tqdm.write(f"    Early stop at epoch {epoch+1}  best_val_loss={best_val:.6f}")
                    break

        total_ep = epoch + 1
        total_t  = time.time() - epoch_start
        tqdm.write(f"    Trained {total_ep} epochs in {total_t:.1f}s  ({total_t/total_ep:.2f}s/ep)")

        # Save training curves
        RESULTS_DIR.mkdir(exist_ok=True)
        curve_path = RESULTS_DIR / f"lstm_curve_{symbol}.csv"
        with open(curve_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
                f.write(f"{i+1},{tl:.6f},{vl:.6f}\n")

        model.load_state_dict(best_state)
        save_lstm(symbol, best_state, scaler, train_end, val_end)

    model.eval()
    context_s = np.concatenate([train_s[-WINDOW:], val_s, test_s])
    X_test, _ = _make_sequences(context_s, WINDOW)
    X_test    = X_test[len(val_s):]   # skip val-period sequences, keep test-period only

    with torch.no_grad():
        preds_s = model(_to_tensor(X_test)).cpu().numpy()

    preds = scaler.inverse_transform(preds_s).flatten()
    return preds, test.values
