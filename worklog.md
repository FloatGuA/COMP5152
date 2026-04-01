---

## #1 · 2026-04-01 16:13 — 讨论项目可行性并完成数据筛选分级脚本

**触发原因**：用户要求（会话结束，触发 work-logger + doc-sync）

### 概述
本次会话围绕 COMP5152 金融时间序列预测项目展开。先讨论了四模型方案（LR/ARIMA/Prophet/LSTM）的可行性和数据需求，确定了 A/B/C/D 四级分类标准（年限 + 行数双维度），然后编写并测试了 filter_datasets.py 脚本。对前 1/5 数据（1609 个文件）跑通，分类结果符合预期。

### 改动清单
- `filter_datasets.py`：新建，实现 CSV 自动分析、分级、复制到 output/ 目录，稀疏文件加 _SPARSE 后缀
- `PROGRESS.md`：新建，记录项目进度
- `TECHNICAL.md`：新建，记录架构设计、数据结构、已知限制

### 决策与背景
- 分级采用双维度（年限 AND 行数）而非单一指标，原因是行数多但年限短的数据缺乏历史广度，年限长但行数少的数据有大缺口，两者需同时满足。
- analyze_csv 不用 pandas，直接读文本首尾行，避免 8000 个文件全量加载内存。
- 先跑前 1/5 验证逻辑，全量再放开。

### 未完成 / 待跟进
- SPARSE 阈值需从 < 1.0 修正为 < 0.9（当前浮点误差导致大量 0.999 被误标）
- 阈值确认后跑全量数据
- 从分级结果中选定目标股票（AAPL / TSLA / SPY），进入建模阶段

---

## #2 · 2026-04-01 17:26 — 搭建建模框架，完成首次 6×4 实验，定位评估问题

**触发原因**：用户要求（/compact 前留痕）

### 概述
搭建了完整的建模框架（src/ 下 6 个模块），随机抽取 6 只 A 级资产（CLI/ALCO/ACCO/DWM/CHII/CEZ），跑通 6×4 首次实验。发现 LR MAPE 虚低（target 设计缺陷）、ARIMA/Prophet fixed-origin 评估失真等问题，制定了 4 个修复任务写入 todolist。确定实验设计：7:2:1 时序切分，MAPE 主指标，1/n 等权平均。

### 改动清单
- `src/data_loader.py`：新建，加载 Close 序列，7:2:1 时序切分
- `src/evaluator.py`：新建，MAPE/RMSE/MAE
- `src/linear_model.py`：新建，lag 特征 + LinearRegression（target 待修复）
- `src/arima_model.py`：新建，auto_arima fixed-origin（待修复为 rolling）
- `src/prophet_model.py`：新建，Prophet fixed-origin（待修复为 rolling）
- `src/lstm_model.py`：新建，PyTorch 双层 LSTM，teacher-forcing 评估
- `run_experiment.py`：新建，6×4 实验主流程，输出 results/metrics.csv
- `PROGRESS.md`：更新已完成、待处理、变更记录
- `TECHNICAL.md`：全面更新，补充建模框架架构、模块说明、设计决策

### 决策与背景
- LSTM 用 PyTorch 而非 TensorFlow：Python 3.14 不支持 TF，PyTorch 2.10 可用。
- ARIMA/Prophet refit 间隔定为 20/30 步（而非用户 prompt 里的 5/10 步）：ALCO 测试集有 1184 行，逐步 refit 预计耗时 20 分钟以上，间隔 refit 是合理折衷。
- 评估统一用 MAPE：不同股票价格量纲差异大，RMSE/MAE 不可直接跨股票平均。

### 未完成 / 待跟进
- Task 1：修复 LR target → next-day Close
- Task 2：修复 ARIMA → rolling one-step，每 20 步 refit
- Task 3：修复 Prophet → rolling one-step，每 30 步 refit
- Task 4：新增统一评估模块（data_group 维度 + composite score + 可视化）
- 修复完成后重跑 run_experiment.py 收集正确指标

---

## #3 · 2026-04-01 23:57 — 修复三个评估问题，新增缓存/进度条/可视化，完成首次正式实验

**触发原因**：用户要求（doc-sync + work-logger）

### 概述
完成了上次留下的 Task 1-3：修复 LR next-day target、ARIMA/Prophet rolling one-step 预测。跑通首次正式实验，ARIMA 均值 MAPE 1.40% 表现最优，LR 2.79% 紧随其后，Prophet 31.62%，LSTM 因 CEZ 异常拉高至 114.25%。发现 CEZ 在 Prophet/LSTM 上出现灾难性 MAPE（110%/558%），已替换为 BND。本次还新增了模型缓存、tqdm 进度条和可视化脚本。

### 改动清单

**模型修复**
- `src/linear_model.py`：target 改为 next-day Close（shift(-1)），集成缓存逻辑
- `src/arima_model.py`：改为 rolling one-step + 每 20 步 refit，集成缓存，新增 tqdm
- `src/prophet_model.py`：改为 rolling one-step + 每 30 步 refit，集成缓存，新增 tqdm
- `src/lstm_model.py`：新增 epoch 级 tqdm（train/val loss + patience），保存训练曲线到 results/，集成缓存

**新增模块**
- `src/model_cache.py`：新建，joblib/torch 持久化，cache key = train_end+val_end 日期
- `plot_results.py`：新建，5 张可视化图（MAPE bar/heatmap/grouped、训练时间、LSTM 曲线）

**实验与数据**
- `run_experiment.py`：新增 tqdm 总进度条，阶段性打印
- `src/data_loader.py`：CEZ 替换为 BND

**文档**
- `PROGRESS.md`：更新已完成、待处理、变更记录
- `TECHNICAL.md`：更新架构图、新增 model_cache 模块说明、缓存文件结构、设计决策

### 决策与背景
- cache key 用日期而非数据 hash：hash 需读全量数据，日期检查开销极低，且切分比例固定时日期足以唯一标识训练集。
- LSTM 缓存同时保存 state_dict 和 scaler：两者必须配套，单独保存 state_dict 在反归一化时会出错。
- CEZ 替换为 BND（美国债券 ETF）：BND 价格区间窄（69-88）、行为稳定，不容易触发 scaler 越界问题。CEZ 异常根因未深查（可能是 MinMaxScaler 超出训练范围），后续如需复盘可单独排查。

### 未完成 / 待跟进
- 用 BND 替换 CEZ 后需重跑实验收集正确指标（当前 metrics.csv 含 CEZ 数据）
- Task 4：统一评估模块（data_group 1yr/5yr/full + composite score + heatmap）
- LSTM 训练曲线（lstm_curve_*.csv）本次实验未生成，因为跑的是旧代码；重跑后会有
- 最终报告撰写
