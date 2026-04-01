import numpy as np
import pandas as pd
import logging
import warnings
from prophet import Prophet
from tqdm import tqdm
from src.model_cache import is_valid, save_pkl, load_pkl

REFIT_EVERY = 30
MODEL_NAME  = "Prophet"


def _fit_prophet(series: pd.Series) -> Prophet:
    df = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)
    m  = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.add_country_holidays(country_name="US")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)
    return m


def predict(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    symbol    = train.name or "unknown"
    train_end = str(train.index[-1].date())
    val_end   = str(val.index[-1].date())
    history   = pd.concat([train, val])

    if is_valid(symbol, MODEL_NAME, train_end, val_end):
        tqdm.write(f"    [cache hit] Prophet {symbol}")
        model = load_pkl(symbol, MODEL_NAME)
    else:
        tqdm.write(f"    Fitting initial Prophet on {len(history)} obs ...")
        model = _fit_prophet(history)
        save_pkl(symbol, MODEL_NAME, model, train_end, val_end)

    tqdm.write(f"    Rolling {len(test)} steps (refit every {REFIT_EVERY})")

    preds      = []
    test_vals  = test.values
    test_dates = test.index

    for i in tqdm(range(len(test_vals)), desc="    Prophet rolling", ncols=80,
                  leave=False, unit="step"):
        future = pd.DataFrame({"ds": [test_dates[i]]})
        pred   = model.predict(future)["yhat"].iloc[0]
        preds.append(pred)

        new_obs = pd.Series([test_vals[i]], index=[test_dates[i]])
        history = pd.concat([history, new_obs])

        if (i + 1) % REFIT_EVERY == 0:
            model = _fit_prophet(history)

    return np.array(preds), test_vals
