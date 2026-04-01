# COMP5152 — Technical Notes

## 架构概览

项目分为两个阶段：数据准备阶段（已完成）和建模阶段（进行中）。

```
archive/
  etfs/      (2165 个 CSV)   ──┐
  stocks/    (5884 个 CSV)   ──┤──> filter_datasets.py ──> output/{A,B,C,D}/{etfs,stocks}/
                                                                    │
                                        从 A 级随机抽取 6 只 ◄────────┘
                                        CLI/ALCO/ACCO/DWM/CHII/BND
                                                    │
                                        src/data_loader.py（加载 + 7:2:1 切分）
                                                    │
                             ┌──────────┬───────────┼───────────┬──────────┐
                         LR  │      ARIMA│       Prophet│       LSTM│
                             │           │              │           │
                        src/linear_model  arima_model  prophet_model lstm_model
                             │           │              │           │
                             └─────────── src/model_cache.py ───────┘
                                          (models/ 目录持久化)
                                                    │
                                        src/evaluator.py（MAPE/RMSE/MAE）
                                                    │
                                        run_experiment.py（汇总 + results/metrics.csv）
                                                    │
                                        plot_results.py（5 张可视化图）
```

## 模块说明

### 数据筛选（filter_datasets.py）

- 职责：读取 archive/ 下的所有 CSV，按年限+行数双维度分级（A/B/C/D），复制到 output/ 对应目录。
- 关键实现：
  - `analyze_csv`：不用 pandas，直接读文本首尾行，避免加载全量数据。
  - `get_grade`：两个条件同时满足才晋级（AND 逻辑）。
  - 前 1/5 测试模式：`all_files[: len(all_files) // 5]`。

### 数据加载（src/data_loader.py）

- 职责：加载指定股票的 Close 序列，按 7:2:1 时序切分（不 shuffle）。
- `SELECTED`：硬编码 6 只资产（CLI/ALCO/ACCO/DWM/CHII/BND），随机抽自 A 级。CEZ 因 Prophet/LSTM 异常 MAPE 已替换为 BND。

### 模型层（src/*_model.py）

每个模型模块暴露统一接口：

```python
def predict(train: pd.Series, val: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    # 返回 (predictions, actuals)，均为 test 集
```

所有模型均集成缓存逻辑：命中缓存时跳过训练，直接加载已有模型。

- **linear_model**：构造 lag_1/5/20、MA7/MA30、ret1d/ret5d 共 7 个特征，StandardScaler 标准化，sklearn LinearRegression。target = next-day Close（`shift(-1)`）。
- **arima_model**：pmdarima auto_arima（d=1，stepwise），rolling one-step 预测，每 20 步全量 refit（`auto_arima` 重新拟合扩展历史）。步间用 `model.update([true_val])` 增量更新。
- **prophet_model**：Facebook Prophet，weekly+yearly 季节性，US 节假日，rolling one-step 预测，每 30 步全量 refit。
- **lstm_model**：PyTorch 实现，LSTM(50)×2 + Dropout(0.2)×2 + Dense(1)，60 天滑动窗口，MinMaxScaler，EarlyStopping（patience=10），teacher-forcing 式 test 评估。训练过程记录 loss 曲线到 `results/lstm_curve_{symbol}.csv`。

### 模型缓存（src/model_cache.py）

- 职责：提供跨运行的模型持久化，避免重复训练。
- cache key：`train_end 日期 + val_end 日期`（日期变化自动失效）。
- 存储路径：`models/{symbol}_{model_name}.pkl`（joblib）或 `models/{symbol}_LSTM.pt`（torch）。
- 接口：`is_valid / save_pkl / load_pkl / save_lstm / load_lstm`。
- LSTM 缓存内容：`state_dict + MinMaxScaler`（两者需同时加载才能正确反归一化）。

### 评估（src/evaluator.py）

- 计算 MAPE、RMSE、MAE，`compute_all` 返回 dict。
- MAPE 是跨股票聚合的主指标（无量纲，1/n 等权平均有意义）。

### 实验主流程（run_experiment.py）

- 遍历 6 只资产 × 4 个模型，记录指标和耗时。
- tqdm 总进度条（24 task），每模型打印 fitting / evaluating / done 阶段及指标。
- 输出 `results/metrics.csv`，打印各模型均值 MAPE 和平均训练时间。

### 可视化（plot_results.py）

- 读取 `results/metrics.csv` 和 `results/lstm_curve_*.csv`。
- 输出 5 张图到 `results/`：
  - `fig_mape_bar.png`：各模型均值 MAPE 柱状图
  - `fig_mape_heatmap.png`：symbol × model MAPE 热力图
  - `fig_mape_grouped.png`：per-symbol 分组柱状图
  - `fig_time_bar.png`：各模型均值训练时间柱状图
  - `fig_lstm_curves.png`：各 symbol 的 LSTM train/val loss 曲线

## 数据结构

### 原始 CSV 格式

```
Date,Open,High,Low,Close,Adj Close,Volume
1980-12-12,0.513,0.515,0.513,0.513,0.406,117258400
```

- 日频数据，Date 格式 `YYYY-MM-DD`，每年约 252 个交易日（美股）。

### 分级标准

```python
GRADE_THRESHOLDS = [
    ("A", min_years=10, min_rows=2500),
    ("B", min_years=5,  min_rows=1250),
    ("C", min_years=3,  min_rows=750),
    # D: 其余所有
]
density = rows / (year_span * 252)  # < 0.9 标记为 SPARSE
```

### metrics.csv 结构

```
category, symbol, model, mape, rmse, mae, train_time_sec, test_rows
```

### 模型缓存文件结构

```
models/
  {symbol}_LinearRegression.pkl   # (scaler, model) tuple，joblib
  {symbol}_ARIMA.pkl              # pmdarima AutoARIMA 对象，joblib
  {symbol}_Prophet.pkl            # Prophet 对象，joblib
  {symbol}_LSTM.pt                # {"state_dict": ..., "scaler": ...}，torch.save
  {symbol}_{model}_meta.json      # {"train_end": "YYYY-MM-DD", "val_end": "YYYY-MM-DD"}
```

## 设计决策

- **LSTM 用 PyTorch 而非 TensorFlow/Keras**：Python 3.14 暂不支持 TensorFlow，PyTorch 2.10 可用。
- **不用 pandas 读 CSV（filter 阶段）**：8000 个文件只需首尾日期，全量加载浪费内存。
- **双维度分级（年限 AND 行数）**：单维度无法同时保证历史广度和训练数据量。
- **复制而非移动**：保留原始文件，便于调整阈值重跑分级。
- **MAPE 为主指标**：价格量纲不同（几元到几百元），RMSE/MAE 不可直接跨股票平均，MAPE 无量纲。
- **ARIMA/Prophet refit 间隔（20/30步）**：完全逐步 refit 在大测试集（ALCO 1184行）上太慢，间隔 refit 是速度与准确性的折衷。
- **cache key 用日期而非数据 hash**：hash 计算需读全量数据，日期检查开销极低，实际使用中切分比例固定，日期足够唯一。
- **LR target 为 next-day Close**：lag_1 特征 ≈ 当天 Close，若 target 也用当天 Close 则模型近似预测自身输入，MAPE 虚低。改为 next-day 才是真正的预测任务。

## 已知限制与可改进方向

**必须改（已有 Task）**：
- Task 4：缺乏 data_group 维度（1yr/5yr/full），无法分析数据量对模型的影响。待新增统一评估模块（composite score + heatmap）。

**如果有时间可以改**：
- SPARSE 阈值当前为 `< 1.0`，浮点误差导致误报，应改为 `< 0.9`。
- ALCO 有 11835 行（1973-2020），历史跨度过长，早期市场结构与现代差异大，可考虑截取最近 N 年。
- LSTM batch size 在 ALCO 这类大数据集上可加大到 64 缩短训练时间。
- filter_datasets.py 可输出 summary.csv 方便后续选股，无需人工翻目录。
- Prophet rolling 每步需重新预测单点，单点 predict 调用开销大，可批量预测后插值优化速度。
- CEZ 被替换的根本原因未确认（MinMaxScaler 超出训练范围 vs 资产本身结构性问题），若后续加回需先排查。
