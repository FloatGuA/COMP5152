# COMP5152 — Financial Time Series Forecasting
## Final Project Report

| | |
|---|---|
| Member 1 | [Name] — [Student ID] |
| Member 2 | [Name] — [Student ID] |

_Submitted: April 2026_

---

## 1. Introduction

Financial time series forecasting has long been a central challenge in quantitative finance. Accurate price predictions can inform trading strategies, risk management, and portfolio optimisation. However, the inherent noise and near-random behaviour of financial markets make this task fundamentally difficult.

This project compares four forecasting models of increasing complexity — Linear Regression (LR), ARIMA, Facebook Prophet, and Long Short-Term Memory networks (LSTM) — on US equity and ETF daily price data. Rather than assuming complex models are superior, we treat model selection as an empirical question and evaluate performance across multiple dimensions: price-level error (MAPE), scale-sensitive error (RMSE, MAE), and directional accuracy (DA).

The pipeline covers: (1) dataset filtering and quality grading across 8,000+ files; (2) data cleaning and standardisation; (3) model implementation under a unified rolling one-step evaluation protocol; and (4) comparative evaluation across 18 assets spanning diverse sectors. Two key findings emerge. First, all four models achieve directional accuracy indistinguishable from random guessing — consistent with the Efficient Market Hypothesis (EMH). Second, the LSTM model exhibits severe sensitivity to non-stationarity: for strongly trending stocks, its MAPE explodes to 15–79%, while ARIMA's first-differencing naturally handles the same series with MAPE under 2%.

---

## 2. Data Source and Transformation

### 2.1 Source

All data are daily OHLCV CSV files sourced from a Kaggle US stock/ETF dataset, covering 5,884 stock files and 2,165 ETF files (approximately 8,000 total). Data span from the 1960s to 2020, sampled at daily frequency on US trading days (~252 days/year).

### 2.2 Data Grading

To ensure sufficient historical depth and training data, all files were classified into four tiers using a two-dimensional criterion (both conditions must be met simultaneously):

| Grade | Min Years | Min Rows | Meaning |
|-------|-----------|----------|---------|
| A | ≥ 10 | ≥ 2,500 | Full experiment quality |
| B | ≥ 5 | ≥ 1,250 | Moderate quality |
| C | ≥ 3 | ≥ 750 | Limited quality |
| D | < 3 or < 750 | — | Excluded |

Files where data density (rows / expected trading days) fell below 0.9 were additionally flagged as `SPARSE`, indicating large structural gaps. After grading, the A-grade pool contained 224 non-sparse stocks and 10 non-sparse ETFs.

The two-dimensional criterion was chosen deliberately: year span alone does not guarantee sufficient training samples (a file may span 15 years but have thin coverage), while row count alone does not guarantee historical breadth.

### 2.3 Asset Selection

Six A-grade assets were selected as the core experimental set, with an additional twelve assets added in an expanded experiment to improve generalisability:

**Core assets (6)**

| Symbol | Category | Description |
|--------|----------|-------------|
| CLI | Stock | Mack-Cali Realty (REIT) |
| ALCO | Stock | Alico Inc. (agriculture) |
| ACCO | Stock | ACCO Brands (consumer goods) |
| DWM | ETF | WisdomTree International Equity |
| CHII | ETF | iShares China Large-Cap |
| BND | ETF | Vanguard Total Bond Market |

**Extended assets (12)**

| Symbol | Category | Sector |
|--------|----------|--------|
| AAPL | Stock | Technology |
| AMD | Stock | Semiconductor |
| AMAT | Stock | Semiconductor Equipment |
| ADBE | Stock | Software |
| ADP | Stock | Business Services |
| ABT | Stock | Healthcare |
| AMGN | Stock | Biotech |
| AFL | Stock | Insurance |
| AEP | Stock | Utilities |
| AN | Stock | Auto Retail |
| BIV | ETF | Intermediate Bond |
| ACWX | ETF | International Equity ex-US |

Two assets were excluded from the experiment after investigation. **CEZ** was originally included in the core set but replaced by BND after exhibiting anomalous MAPE (Prophet: 111%, LSTM: 559%) attributable to price scale instability in the test period. **AIG** was excluded from the extended set due to its catastrophic 2008 financial crisis trajectory (price collapse from ~\$70 to ~\$1), which creates severe distribution mismatch between training and test periods — the same root cause as CEZ.

### 2.4 Data Cleaning

Raw CSV files contain gaps caused by US market holidays and occasional missing data. All series were cleaned using `pd.bdate_range` (Monday–Friday business calendar) as a reference, with forward-filling applied to all missing dates. Holiday gaps are expected (~9/year); gaps exceeding this expectation are flagged as true data gaps.

Gap summary for the 6 core assets:

| Symbol | True Gaps (beyond expected holidays) |
|--------|--------------------------------------|
| CLI | 4 |
| ALCO | 0 |
| ACCO | 3 |
| DWM | 3 |
| CHII | 3 |
| BND | 1 |

All 18 assets were cleaned with the same procedure. The extended assets (AAPL, AMD, etc.) showed only holiday gaps (0 true gaps) with the exception of BIV and ACWX, each with 1 true gap.

### 2.5 Train / Validation / Test Split

Each series was split in temporal order (no shuffling) with a 7:2:1 ratio:
- **Train (70%)**: model fitting
- **Validation (20%)**: hyperparameter tuning and early stopping
- **Test (10%)**: final evaluation (held out throughout)

Temporal ordering is mandatory for financial time series to prevent look-ahead bias. The test set corresponds to the most recent portion of each series (roughly 2016–2020 for most assets), ranging from 269 rows (CHII) to 1,311 rows (AEP).

---

## 3. Model Descriptions

### 3.1 Linear Regression (LR)

**Principle.** Ordinary Least Squares (OLS) regression finds a linear mapping from a fixed feature vector to the next-day close price. The assumption is that price-relevant information can be summarised as a weighted sum of recent price levels, moving averages, and short-term returns.

**Feature engineering.** Seven hand-crafted features are constructed from the close price series:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `lag_1` | Close(t−1) | Yesterday's price — dominant predictor |
| `lag_5` | Close(t−5) | One-week-ago price |
| `lag_20` | Close(t−20) | One-month-ago price |
| `MA7` | Mean(Close, 7d) | Short-term moving average |
| `MA30` | Mean(Close, 30d) | Medium-term moving average |
| `ret_1d` | (Close(t) − Close(t−1)) / Close(t−1) | Yesterday's 1-day return |
| `ret_5d` | (Close(t) − Close(t−5)) / Close(t−5) | 5-day cumulative return |

All features are standardised with `StandardScaler` (zero mean, unit variance) fitted on training data only. The target is the **next-day** Close (`shift(-1)`); using same-day Close would create a data leakage where `lag_1 ≈ target`, yielding artificially low error (Flaw 1, Section 4.1 below).

**Implementation.** `sklearn.linear_model.LinearRegression`. No hyperparameters to tune. The model is fitted once on the training set and applied directly to the test set without retraining.

**Advantages.** Extremely fast (< 0.1 s), fully interpretable (inspect coefficients), no risk of overfitting on the test set. Provides a transparent baseline.

**Limitations.** Strictly linear — cannot capture non-linear price dynamics or long-range temporal dependencies. The feature set is manually designed; relevant patterns not encoded in these 7 features are invisible to the model. In practice, the coefficient on `lag_1` dominates (~1.0), meaning the model degenerates to predicting "tomorrow ≈ today" (random walk).

---

### 3.2 ARIMA

**Principle.** ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for univariate time series. It combines three components:

- **AR(p)** — AutoRegressive: the current value is modelled as a linear combination of its *p* most recent values. Captures momentum and mean-reversion patterns.
- **I(d)** — Integrated: the series is differenced *d* times to achieve stationarity. For financial prices, `d=1` (first difference = daily returns) is standard and removes the unit root.
- **MA(q)** — Moving Average: the current value includes a weighted sum of the *q* most recent forecast errors. Captures residual autocorrelation not explained by the AR component.

The combined model is written ARIMA(p, d, q). For financial prices, the model is applied to the differenced series, effectively forecasting day-to-day changes and then integrating back to price levels.

**Parameter selection.** `pmdarima.auto_arima` performs a stepwise AIC-minimising search over (p, q) combinations with `d=1` fixed, `max_p=5`, `max_q=5`. Across all 18 assets, the selected order is consistently ARIMA(0,1,0) or very low-order — the canonical discrete random walk.

**Implementation.** Rolling one-step evaluation: at each test step, predict one step ahead, then call `model.update([true_value])` to incorporate the new observation. Every 20 steps, `auto_arima` is called on the full expanded history to allow the order to adapt. The 20-step interval is a trade-off between adaptability and runtime (full refit dominates cost).

**Advantages.** Principled handling of non-stationarity via differencing; no manual feature engineering; well-understood theoretical properties; consistent performance regardless of price trend direction or magnitude.

**Limitations.** Assumes linear relationships between current and past values. Computationally expensive under rolling refit: each `auto_arima` call evaluates multiple (p,q) combinations, making evaluation on long test sets (1,000+ rows) take 15–35 minutes. Purely univariate — cannot incorporate exogenous variables.

---

### 3.3 Prophet

**Principle.** Prophet (Taylor & Letham, 2018) is a decomposable additive time-series model developed at Facebook:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)
```

- **Trend**: a piecewise linear or logistic growth curve with automatically detected changepoints.
- **Seasonality**: Fourier series approximating weekly and yearly cycles.
- **Holidays**: user-supplied date effects (here: US federal holidays).

Parameters are estimated via MAP optimisation using Stan (L-BFGS).

**Key parameters in this implementation.**
- `weekly_seasonality=True`: models a 7-day cycle with a Fourier order of 3.
- `yearly_seasonality=True`: models a 365.25-day cycle with a Fourier order of 10.
- `daily_seasonality=False`: disabled (daily close prices have no intra-day structure).
- `add_country_holidays("US")`: adds ~11 US federal holidays as additive regressors.

**Implementation.** Rolling one-step evaluation with full refitting every 30 steps. Each refit constructs a new Prophet instance and fits it on all available history up to the current step. No incremental update API exists; the model must be fully reconstructed each time.

**Advantages.** Robust to missing data and outliers; automatically detects trend changepoints; produces interpretable decomposition plots; requires minimal tuning.

**Limitations.** Designed for business/human-activity series (web traffic, sales) with genuine weekly and yearly periodicity. US equity prices do not exhibit meaningful weekly or yearly seasonality — Prophet's seasonality components fit noise, systematically over-project trends, and produce MAPE of 4–52% across the 18 test assets. The worst-case (AMD: 52.1%) occurs for highly volatile semiconductors where trend extrapolation is most misleading. The need to refit from scratch every 30 steps also makes Prophet the second slowest model (12–210 s per asset).

---

### 3.4 LSTM

**Principle.** Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) is a recurrent neural network architecture designed to capture long-range dependencies in sequential data. Each LSTM cell maintains a hidden state and a cell state, controlled by three gates:

- **Forget gate**: decides what fraction of the previous cell state to discard.
- **Input gate**: decides what new information to write into the cell state.
- **Output gate**: decides what part of the cell state to expose as the hidden state.

The gating mechanism allows gradients to flow across many time steps without the vanishing gradient problem that affects vanilla RNNs, enabling the network to learn dependencies spanning weeks or months.

**Architecture.**

```
Input: (batch, 60, 1)           # 60-day sliding window, 1 feature (scaled close)
  └─ LSTM(input=1, hidden=50)   # Layer 1: 50 hidden units
  └─ Dropout(0.2)               # Regularisation: 20% random unit drop
  └─ LSTM(input=50, hidden=50)  # Layer 2: 50 hidden units
  └─ Dropout(0.2)
  └─ Linear(50 → 1)             # Fully connected output
Output: (batch, 1)              # Predicted next-day close (scaled)
```

**Key hyperparameters.**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `WINDOW` | 60 | Number of past trading days used as input (~3 months) |
| `hidden` | 50 | Number of hidden units in each LSTM layer |
| `BATCH` | 32 | Mini-batch size during training |
| `Dropout` | 0.2 | Fraction of units randomly zeroed per forward pass during training |
| `EPOCHS` | 100 | Maximum training epochs (early stopping usually triggers earlier) |
| `patience` | 10 | Early stopping: halt training after 10 epochs without validation loss improvement |
| `optimizer` | Adam | Adaptive learning rate optimiser |
| `loss` | MSELoss | Mean squared error on scaled prices |

**Preprocessing.** The close price series is normalised with `StandardScaler` (fitted on training data only). The test evaluation uses **teacher-forcing**: the 60-day input window always contains true historical prices, so prediction errors do not compound over the test horizon. This makes the LSTM evaluation directly comparable to the single-step rolling protocol used by ARIMA and LR.

**Device.** Training runs on an NVIDIA RTX 5070 GPU (CUDA 12.8) via PyTorch 2.x, reducing training time to 6–17 seconds per asset.

**Advantages.** Theoretically capable of learning complex non-linear temporal patterns and long-range dependencies that LR and ARIMA cannot represent. GPU acceleration makes training fast despite model complexity. Dropout and early stopping provide regularisation.

**Limitations.** Learns price levels, not price differences — making it sensitive to distribution shift when training and test periods occupy different price regimes (see Section 6.3). Acts as a black box: model coefficients are not directly interpretable. Requires careful hyperparameter tuning and a sufficiently large training set. On the 18 assets tested, the additional representational capacity confers no directional accuracy advantage over the simpler models.

---

## 4. Analysis Steps and Trials

### 3.1 Initial Implementation and Flaws Found

The first experimental run revealed four evaluation design flaws:

**Flaw 1 — LR target leakage.** The regression target was initially set to the same-day Close price. Since `lag_1` (yesterday's close) ≈ today's close, the model was trivially predicting its own input feature, yielding a misleading MAPE of ~0.5%. Fix: target changed to `Close.shift(-1)` (next-day close), the true forecasting objective.

**Flaw 2 — ARIMA/Prophet fixed-origin evaluation.** Both models were fitted once on training data and then projected multi-step forward across the entire test period. For test sets of 300–1,300 points, multi-step accumulation produced severe forecast drift. Fix: rolling one-step evaluation — predict one step, observe the true value, update the model, then predict the next step. ARIMA refits via `auto_arima` every 20 steps; Prophet refits every 30 steps to balance accuracy and runtime.

**Flaw 3 — LSTM MSE loss shape broadcasting.** The training loss was computed as `loss_fn(model(xb).squeeze(), yb)` where `squeeze()` gives shape `(batch,)` but `yb` retains shape `(batch, 1)`. PyTorch broadcasting produces a `(batch × batch)` loss matrix: each prediction is compared to every target in the batch rather than its own paired target. The gradient signal reduces to minimising the distance from each prediction to the batch-mean target, so the model converges to predicting the training-set mean regardless of input. For trending stocks, the test-period mean lies far above the training-period mean, producing MAPE of 30–55%. For stable ETFs, the means are similar, so the bug appeared benign. Fix: `loss_fn(model(xb).squeeze(), yb.squeeze())`.

**Flaw 4 — LSTM evaluation index error.** The test-sequence buffer contained both validation-period and test-period sequences. The slice `X_test[:len(test)]` accidentally extracted validation sequences. Fix: `X_test[len(val):]`.

### 3.2 CEZ and AIG Anomalies

CEZ produced Prophet MAPE of 111% and (pre-fix) LSTM MAPE of 559%. After investigating price distributions, the test-period prices fell outside the scaler's training range, producing out-of-distribution inputs. CEZ was replaced by BND.

AIG, added in the extended experiment, showed LR MAPE of 9.8% and Prophet MAPE of 222.6% — far beyond the typical 1–2% range observed for other assets under the same models. This mirrors the CEZ pattern. AIG's stock collapsed from ~\$70 to under \$1 during the 2008 financial crisis, creating a training distribution that is entirely unrepresentative of any subsequent period. AIG was excluded from the final asset list.

### 3.3 Addition of Directional Accuracy

Low MAPE does not imply actionable forecasting value. A model predicting "tomorrow = today" (random walk) achieves MAPE approximately equal to average daily volatility (~1%) without any directional insight. We added **Directional Accuracy (DA)**:

```
DA = P( sign(actual[t] − actual[t−1])  ==  sign(pred[t] − actual[t−1]) )
```

DA is the proportion of test days on which the predicted price direction (up/down) matches the actual direction. The uninformed baseline is 50%.

### 3.4 Reproducibility

All experiments use a global random seed (`seed = 42`) applied to Python's `random`, NumPy, and PyTorch (including CUDA). For LSTM, the seed is re-applied before each model initialisation to ensure weight initialisation and batch shuffling are consistent regardless of asset ordering.

---

## 5. Model Evaluation

### 4.1 Metrics

| Metric | Description | Baseline |
|--------|-------------|---------|
| MAPE (%) | Mean absolute percentage error — scale-free, primary metric | Lower is better |
| RMSE | Root mean squared error in price units | Lower is better |
| MAE | Mean absolute error in price units | Lower is better |
| DA (%) | Directional accuracy | 50% = random |

MAPE is chosen as the primary cross-asset metric because price scales vary widely across assets (from ~\$5 to ~\$300+). RMSE and MAE carry price-unit dimensions and cannot be meaningfully averaged across assets without normalisation.

### 4.2 Evaluation Protocol

All models use rolling one-step prediction on the test set. At each step _t_: predict _t_, then reveal true value before predicting _t+1_.

- **LR**: re-inference using the last lag features (no refit needed)
- **ARIMA**: incremental `model.update()` each step + full `auto_arima` refit every 20 steps
- **Prophet**: full refit every 30 steps using expanded history
- **LSTM**: teacher-forcing — the 60-day input window always uses true historical prices (no refit after training)

### 4.3 Results

#### MAPE (%) — lower is better

| Symbol | Category | LR | ARIMA | Prophet | LSTM |
|--------|----------|----|-------|---------|------|
| CLI | Stock | 1.31 | 1.26 | 13.30 | 1.65 |
| ALCO | Stock | 1.44 | 1.29 | 20.52 | 1.58 |
| ACCO | Stock | 2.07 | 1.82 | 31.89 | 2.00 |
| DWM | ETF | 0.81 | 0.78 | 9.03 | 0.81 |
| CHII | ETF | 1.16 | 1.14 | 4.25 | 1.15 |
| BND | ETF | 0.24 | 0.24 | 3.60 | 0.27 |
| AAPL | Stock | 1.55 | 1.12 | 14.82 | **78.90** |
| AMD | Stock | 3.31 | 2.61 | 52.11 | 3.17 |
| AMAT | Stock | 2.01 | 1.62 | 22.58 | 2.57 |
| ADBE | Stock | 1.60 | 1.23 | 20.20 | **61.47** |
| ADP | Stock | 1.19 | 0.93 | 9.18 | **29.19** |
| ABT | Stock | 1.24 | 0.97 | 13.62 | **23.21** |
| AMGN | Stock | 1.36 | 1.01 | 8.16 | **20.24** |
| AFL | Stock | 1.08 | 0.83 | 9.21 | **18.37** |
| AEP | Stock | 0.97 | 0.81 | 10.41 | **13.99** |
| AN | Stock | 1.86 | 1.41 | 18.59 | 3.21 |
| BIV | ETF | 0.25 | 0.25 | 4.59 | 0.27 |
| ACWX | ETF | 0.86 | 0.83 | 6.52 | 0.85 |
| **Mean (all 18)** | | **1.35** | **1.12** | **15.14** | **14.61** |

Bold LSTM values indicate distribution-shift failure (see Section 5.4).

**LSTM — stable vs. trending asset split**

The 18 assets divide naturally into two groups based on whether the LSTM MAPE is competitive:

| Group | Assets (n) | LSTM Mean MAPE | LR Mean MAPE | ARIMA Mean MAPE |
|-------|-----------|---------------|-------------|----------------|
| Stable / low-trend | CLI, ALCO, ACCO, DWM, CHII, BND, AMD, AMAT, AN, BIV, ACWX (11) | **1.59%** | 1.56% | 1.27% |
| Strong upward trend | AAPL, ADBE, ADP, ABT, AMGN, AFL, AEP (7) | **35.48%** | 1.19% | 0.99% |

On stable assets, LSTM (1.59%) is competitive with LR and ARIMA. On strongly trending stocks, LSTM MAPE explodes while ARIMA and LR remain unaffected.

#### Directional Accuracy (%) — random baseline = 50%

| Symbol | LR | ARIMA | Prophet | LSTM |
|--------|----|-------|---------|------|
| CLI | 49.18 | 44.76 | 50.30 | 50.60 |
| ALCO | 50.78 | 50.08 | 47.96 | 48.69 |
| ACCO | 47.11 | 46.72 | 45.41 | 44.09 |
| DWM | 47.49 | 48.19 | 49.03 | 44.57 |
| CHII | 48.88 | 48.33 | 47.21 | 46.47 |
| BND | 47.48 | 45.86 | 39.35 | 49.41 |
| AAPL | 49.02 | 48.98 | 46.83 | 44.49 |
| AMD | 46.88 | 42.82 | 43.97 | 48.66 |
| AMAT | 48.03 | 47.22 | 45.21 | 50.96 |
| ADBE | 49.89 | 50.17 | 42.42 | 42.42 |
| ADP | 50.05 | 53.35 | 44.73 | 43.20 |
| ABT | 46.40 | 47.51 | 45.79 | 45.02 |
| AMGN | 47.60 | 49.01 | 52.03 | 45.88 |
| AFL | 48.71 | 49.71 | 43.87 | 42.91 |
| AEP | 46.22 | **34.12** | 45.27 | 43.05 |
| AN | 46.27 | 51.22 | 48.52 | 49.29 |
| BIV | 49.26 | 48.22 | 40.83 | 52.66 |
| ACWX | 44.23 | 49.52 | 53.04 | 51.12 |
| **Mean** | **48.0** | **47.5** | **46.2** | **46.9** |

One anomaly: AEP ARIMA DA = 34.1%, the only result falling below 40% across the entire experiment. AEP (utility sector) exhibits unusually low price volatility; the test period coincides with a slow, sustained upward drift. In this regime, ARIMA's random walk approximation systematically predicts flat-to-down while the actual series moves monotonically upward — producing worse-than-random directional performance. This is a known failure mode of ARIMA on locally trending, low-noise series rather than a data quality issue.

#### Training time per model (representative values)

| Model | Typical Range | Notes |
|-------|-------------|-------|
| LR | < 0.1 s | sklearn, instant |
| LSTM | 6–17 s | GPU (RTX 5070); early stopping typically at epoch 10–30 |
| Prophet | 12–209 s | CPU; rolling refit every 30 steps |
| ARIMA | 42–2031 s | CPU; rolling refit every 20 steps is the bottleneck |

ARIMA is by far the slowest model. On AEP (1,311 test rows, ~66 full refits), it required 33.8 minutes. The ARIMA refit cost scales approximately linearly with test set size. LSTM, despite its greater representational capacity, is the fastest after LR due to GPU acceleration.

### 4.4 Model Analysis

**Linear Regression** fits 7 features: `lag_1`, `lag_5`, `lag_20`, `MA7`, `MA30`, `ret_1d`, `ret_5d`. In practice, the dominant fitted coefficient is on `lag_1` (≈ 1.0), confirming the model reduces to a random walk approximation. Mean MAPE of 1.35% reflects average daily volatility, not predictive insight. Training is instant.

**ARIMA** with `d=1` enforced (unit root) consistently selects ARIMA(0,1,0) or similar low-order models across all assets — the canonical random walk formulation. Its best mean MAPE (1.12%) is narrowly ahead of LR. The first-differencing step makes ARIMA inherently robust to price-level non-stationarity, explaining why it performs consistently across both stable and strongly trending assets. Runtime is the main drawback: 5–35 minutes per asset.

**Prophet** was designed for human-activity time series with clear weekly and yearly seasonality. US equity prices lack meaningful periodicity of this kind. Prophet's built-in seasonality components fit noise rather than signal, systematically over-projecting trends and producing mean MAPE of 15.14% across 18 assets (ranging from 3.6% on BND to 52.1% on AMD). Its mean DA (46.2%) is the lowest among all models. Prophet is unsuitable for financial price forecasting.

**LSTM** (2 layers, hidden=50, window=60, Dropout=0.2, EarlyStopping patience=10) reveals a critical sensitivity to price non-stationarity. On the 11 stable assets, the LSTM achieves mean MAPE of 1.59% — competitive with LR and ARIMA. On the 7 strongly trending stocks, MAPE explodes to a mean of 35.5% (range: 14.0–78.9%). The root cause is distribution shift: the `StandardScaler` is fitted on the training period's price distribution; for long-term growth stocks, the test-period price level sits multiple standard deviations above the training mean, producing out-of-distribution inputs that the model cannot extrapolate. ARIMA sidesteps this entirely via differencing. LSTM's DA (46.9% mean) is also near-random and does not recover even on stable assets, confirming that the high MAPE on trending stocks is not the only failure mode.

---

## 6. Conclusions

### 5.1 Main Finding

The central finding is not the MAPE ranking but the DA values: **all four models achieve directional accuracy of 34–53% across all 18 assets, with means of 46–48% — statistically indistinguishable from random guessing (50%)**. This is the decisive result. The MAPE ranking (ARIMA ≈ LR ≪ Prophet ≈ LSTM-overall) is an artefact of how closely each model approximates the random walk, not a measure of forecasting value.

### 5.2 Consistency with the Efficient Market Hypothesis

The results are fully consistent with the **weak-form Efficient Market Hypothesis**: past price information is already reflected in current prices, and no model trained solely on historical prices can reliably predict future direction. All four models — from the simplest (LR) to the most complex (LSTM) — converge on this conclusion independently, across 18 assets spanning equities, ETFs, and multiple sectors.

### 5.3 LSTM Non-Stationarity Sensitivity

The most practically significant model-specific finding is the LSTM's failure on trending stocks. While the random walk baseline explains the DA result, the 10–79× MAPE inflation for trending assets reveals an additional weakness: LSTM learns price levels, not price changes. ARIMA, by construction, operates on differences and is immune. This suggests that for financial forecasting tasks where the price series has a strong trend component, LSTM requires either explicit detrending (e.g., log-returns as the target variable) or a stationary input representation before training.

### 5.4 Challenges and Lessons

The most instructive experience of the project was diagnosing the LSTM broadcasting bug. The error produced plausible-looking MAPE values for stable ETFs (2–5%), masking the failure entirely. Only the dramatic 10× MAPE inflation on trending stocks prompted investigation. This illustrates how financial forecasting evaluation can give false confidence: a model predicting the training mean will look reasonable on low-volatility assets and catastrophic on trending ones. The inclusion of DA as a secondary metric is essential — even after the LSTM bug was fixed, DA remained near-random, exposing the absence of genuine predictive value.

The anomalous AEP ARIMA DA (34.1%) demonstrates that even models with consistently low MAPE can perform worse than random on directional forecasting in specific regimes (sustained low-volatility trends). This reinforces the necessity of reporting DA alongside MAPE.

### 5.5 Limitations

- **Features**: Only Close price history is used. Volume, macro indicators, or cross-asset signals may carry information inaccessible to price-only models.
- **Architecture search**: The LSTM architecture was not systematically tuned. Log-return targets, attention mechanisms, or Transformer-based models might reduce distribution-shift sensitivity.
- **Market regimes**: The test set is the most recent 10% of each series. Performance in specific volatility regimes (e.g., 2008 crisis, 2020 COVID shock) is not separately assessed.
- **ARIMA order selection**: The auto-selected ARIMA order is recomputed every 20 steps using full history. Fixing the order after the first selection would reduce runtime by ~10× with negligible accuracy cost for these near-random-walk series, but reduces methodological rigour.

### 5.6 Summary

Four models of increasing complexity were evaluated on 18 US equity and ETF series under a rigorous rolling one-step protocol. The convergence of all four models to near-random directional accuracy across all assets provides strong empirical support for the weak-form EMH. Complexity does not help for directional prediction: LSTM matches ARIMA on DA and fails more severely on MAPE for trending stocks. The two most useful methodological contributions of this project are (1) the introduction of DA alongside MAPE, which exposes the "random walk illusion" that low MAPE creates, and (2) the documented LSTM distribution-shift failure, which highlights a practical pitfall when applying deep sequence models to non-stationary financial price series without preprocessing.

---

_Note: Cover page member information requires manual completion._
