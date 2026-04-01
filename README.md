# COMP5152 金融时间序列预测 / Financial Time Series Forecasting

对美股及 ETF 日频收盘价进行四模型对比实验，量化评估 Linear Regression、ARIMA、Prophet、LSTM 在真实金融数据上的预测精度与训练开销。

A four-model comparison experiment on daily close prices of US stocks and ETFs, quantitatively evaluating the prediction accuracy and training cost of Linear Regression, ARIMA, Prophet, and LSTM on real financial data.

## 功能特性 / Features

- 自动对 8000+ 个 CSV 数据集按年限和行数分级（A/B/C/D），筛选高质量资产
  Automatically grades 8000+ CSV datasets by year span and row count (A/B/C/D) to select high-quality assets
- 四模型统一接口，7:2:1 时序切分，结果可直接横向比较
  Four models share a unified interface with 7:2:1 time-ordered split for fair cross-model comparison
- 模型训练结果持久化缓存，换数据集或调参后自动失效重训
  Trained models are cached to disk and automatically invalidated when data or split changes
- tqdm 分级进度条：总任务进度、ARIMA/Prophet rolling 步进度、LSTM epoch 级别（loss + patience）
  Hierarchical tqdm progress bars: overall task, ARIMA/Prophet rolling steps, LSTM per-epoch (loss + patience)
- 一键生成 5 张可视化图：MAPE 柱状图、热力图、分组对比图、训练时间图、LSTM loss 曲线
  One-command visualization: MAPE bar chart, heatmap, grouped comparison, training time, LSTM loss curves

## 安装 / Installation

需要 Python 3.10+（已在 Python 3.14 上验证，TensorFlow 不可用，使用 PyTorch 替代）。

Requires Python 3.10+ (tested on Python 3.14; TensorFlow is not supported, PyTorch is used instead).

```bash
pip install pandas numpy scikit-learn pmdarima prophet torch tqdm matplotlib seaborn joblib
```

## 目录结构 / Project Structure

```
selected_data/          # 选定的实验数据集（6 只资产）
  stocks/               # CLI, ALCO, ACCO
  etfs/                 # DWM, CHII, BND
src/                    # 模型与工具模块
  data_loader.py        # 数据加载与切分
  evaluator.py          # MAPE / RMSE / MAE
  model_cache.py        # 模型持久化缓存
  linear_model.py
  arima_model.py
  prophet_model.py
  lstm_model.py
run_experiment.py       # 实验主入口
plot_results.py         # 可视化
filter_datasets.py      # 数据分级（预处理，仅需跑一次）
results/                # 实验输出（metrics.csv + 图片）
models/                 # 模型缓存（自动生成，可删除重训）
archive/                # 原始数据集（不纳入 git）
```

## 使用方式 / Usage

**第一步：运行实验**

Run the experiment:

```bash
python run_experiment.py
```

遍历 6 只资产 × 4 个模型，输出 `results/metrics.csv`，并在终端打印各模型均值 MAPE 和平均训练时间。第二次运行时已训练的模型会从缓存加载，显著提速。

Iterates over 6 assets × 4 models, writes `results/metrics.csv`, and prints mean MAPE and average training time per model. On subsequent runs, cached models are loaded directly.

**第二步：生成可视化**

Generate visualizations:

```bash
python plot_results.py
```

读取 `results/metrics.csv`，在 `results/` 目录下生成 5 张图片。

Reads `results/metrics.csv` and saves 5 figures to `results/`.

**（可选）重新分级原始数据集**

(Optional) Re-grade the raw dataset archive:

```bash
python filter_datasets.py
```

读取 `archive/` 下所有 CSV，按分级标准复制到 `output/{A,B,C,D}/{etfs,stocks}/`。通常只需跑一次。

Reads all CSVs from `archive/`, grades them, and copies to `output/{A,B,C,D}/{etfs,stocks}/`. Typically only needed once.

## 替换实验资产 / Swapping Assets

编辑 `src/data_loader.py` 中的 `SELECTED` 列表，并将对应 CSV 放入 `selected_data/` 对应子目录，再跑 `run_experiment.py` 即可。旧缓存不影响新资产。

Edit the `SELECTED` list in `src/data_loader.py`, place the corresponding CSV in `selected_data/`, then re-run `run_experiment.py`. Old cache entries do not affect new assets.

## 技术栈 / Tech Stack

- pandas, numpy — 数据处理与特征工程
  Data processing and feature engineering
- scikit-learn — Linear Regression、StandardScaler、MinMaxScaler
  Linear Regression, StandardScaler, MinMaxScaler
- pmdarima — ARIMA 自动参数选择（auto_arima）
  Automatic ARIMA order selection (auto_arima)
- prophet — Facebook Prophet 时间序列模型
  Facebook Prophet time series model
- torch (PyTorch) — 双层 LSTM 实现与训练
  Two-layer LSTM implementation and training
- joblib — 非神经网络模型的序列化缓存
  Serialization cache for non-neural models
- tqdm — 多级进度条
  Multi-level progress bars
- matplotlib, seaborn — 结果可视化
  Result visualization
