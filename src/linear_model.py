import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.model_cache import is_valid, save_pkl, load_pkl

MODEL_NAME = "LinearRegression"


def _make_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": series})
    for lag in [1, 5, 20]:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df["ma7"]    = df["close"].rolling(7).mean()
    df["ma30"]   = df["close"].rolling(30).mean()
    df["ret1"]   = df["close"].pct_change(1)
    df["ret5"]   = df["close"].pct_change(5)
    df["target"] = df["close"].shift(-1)   # next-day Close
    return df.dropna()


def predict(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    symbol    = train.name or "unknown"
    train_end = str(train.index[-1].date())
    val_end   = str(val.index[-1].date())

    full = pd.concat([train, val, test])
    feat = _make_features(full)
    feature_cols = [c for c in feat.columns if c not in ("close", "target")]
    test_feat = feat.loc[feat.index.isin(test.index)]
    X_test    = test_feat[feature_cols].values
    y_test    = test_feat["target"].values

    if is_valid(symbol, MODEL_NAME, train_end, val_end):
        tqdm.write(f"    [cache hit] LR {symbol}")
        scaler, model = load_pkl(symbol, MODEL_NAME)
    else:
        train_feat = feat.loc[feat.index.isin(train.index)]
        X_train = train_feat[feature_cols].values
        y_train = train_feat["target"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_s, y_train)

        save_pkl(symbol, MODEL_NAME, (scaler, model), train_end, val_end)

    X_test_s = scaler.transform(X_test)
    return model.predict(X_test_s), y_test
