import numpy as np
import pandas as pd
import warnings
from pmdarima import auto_arima
from tqdm import tqdm
from src.model_cache import is_valid, save_pkl, load_pkl

REFIT_EVERY = 20
MODEL_NAME  = "ARIMA"


def predict(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    symbol    = train.name or "unknown"
    train_end = str(train.index[-1].date())
    val_end   = str(val.index[-1].date())
    history   = pd.concat([train, val])

    if is_valid(symbol, MODEL_NAME, train_end, val_end):
        tqdm.write(f"    [cache hit] ARIMA {symbol}")
        model = load_pkl(symbol, MODEL_NAME)
    else:
        tqdm.write(f"    Fitting initial ARIMA on {len(history)} obs ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                history.values,
                d=1, max_p=5, max_q=5,
                seasonal=False, stepwise=True,
                error_action="ignore", suppress_warnings=True,
            )
        save_pkl(symbol, MODEL_NAME, model, train_end, val_end)

    tqdm.write(f"    Order: {model.order}  |  rolling {len(test)} steps (refit every {REFIT_EVERY})")

    preds     = []
    test_vals = test.values

    for i in tqdm(range(len(test_vals)), desc="    ARIMA rolling", ncols=80,
                  leave=False, unit="step"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(n_periods=1)[0]
        preds.append(pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.update([test_vals[i]])

        if (i + 1) % REFIT_EVERY == 0:
            history_ext = np.append(history.values, test_vals[: i + 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = auto_arima(
                    history_ext,
                    d=1, max_p=5, max_q=5,
                    seasonal=False, stepwise=True,
                    error_action="ignore", suppress_warnings=True,
                )

    return np.array(preds), test_vals
