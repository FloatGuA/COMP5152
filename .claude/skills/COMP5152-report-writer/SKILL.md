---
name: COMP5152-report-writer
description: 为 COMP5152 项目撰写或更新 report.md。每完成一个阶段后触发，自动从 results/metrics.csv、TECHNICAL.md、PROGRESS.md、worklog.md 读取最新状态，生成或覆盖更新符合课程要求的完整实验报告。当用户说"更新报告"、"写报告"、"同步报告"、"触发 COMP5152-report-writer"或提到要生成/刷新 report.md 时使用。
---

# Report Writer — COMP5152

生成或全量覆盖更新项目根目录下的 `report.md`。报告以当前项目状态为准，每次触发都重新生成完整文件。

---

## 第一步：收集素材

按顺序读取以下文件（存在则读，不存在则跳过并在报告中注明"待补充"）：

1. `results/metrics.csv` — 实验结果（MAPE/RMSE/MAE/训练时间）
2. `TECHNICAL.md` — 架构、模块、设计决策
3. `PROGRESS.md` — 当前进度与待处理事项
4. `worklog.md` — 工作历史与决策背景
5. `src/data_cleaner.py` — 数据清洗逻辑
6. `src/data_loader.py` — 数据集列表（SELECTED）

读完后，在脑中整理：
- 用了哪 6 只资产、哪 4 个模型
- 数据清洗做了什么
- 各模型 MAPE/RMSE/MAE 的具体数字
- 发现了哪些问题、怎么修复的
- 目前还有什么未完成

---

## 第二步：生成 report.md

按以下结构写出完整报告，语言用**英文**，格式清晰、客观、学术风格。

### 封面页

```
# COMP5152 — Financial Time Series Forecasting
## Final Project Report

| | |
|---|---|
| Member 1 | [Name] — [Student ID] |
| Member 2 | [Name] — [Student ID] |

_Submitted: [Date]_
```

成员信息保留占位符，不要填写任何假数据。

---

### 各章节写作要求

**Introduction**
- 说明项目目标：对比 4 种模型在金融时间序列预测上的表现
- 说明研究意义：金融预测的实际价值、模型选择对结果的影响
- 简述整体方法：数据筛选 → 清洗 → 建模 → 评估
- 2-3 段，不超过 300 词

**Data Source and Transformation**
- 数据来源：Kaggle 美股/ETF 日频 OHLCV CSV，覆盖 8000+ 个文件
- 数据分级标准（A/B/C/D）：年限 AND 行数双维度，附具体阈值
- 最终选定的 6 只资产（从 SELECTED 列表读取），说明为何选 A 级
- 数据清洗：用 `pd.bdate_range` 检测并 forward-fill 缺失交易日，列出各资产检测到的 gap 数量
- 切分方式：7:2:1 时序切分（不 shuffle），说明理由
- 如果 data_cleaner.py 里有具体 gap 数字，直接引用

**Analysis Steps and Trials**
- 第一轮实验：发现的三个评估问题（LR target 缺陷、ARIMA/Prophet fixed-origin、LSTM 已是 rolling）
- 修复过程：如何逐一修复，决策依据（如 refit 间隔选 20/30 步的原因）
- 模型缓存机制的引入（为何、如何实现）
- CEZ 异常的发现与替换（如 worklog 中有记录）
- 按时间顺序叙述，体现出迭代试错的过程

**Model Evaluation**
- 评估指标：MAPE（主指标，无量纲，可跨资产平均）、RMSE、MAE，解释为何选 MAPE
- 评估方法：rolling one-step next-day 预测，teacher-forcing（LSTM）
- 结果表格（从 metrics.csv 提取真实数字）：

  | Symbol | LR MAPE | ARIMA MAPE | Prophet MAPE | LSTM MAPE |
  |--------|---------|-----------|-------------|---------|
  | ...    | ...     | ...       | ...         | ...     |
  | **Mean** | ... | ... | ... | ... |

- 训练时间对比表
- 逐模型分析：各模型的表现特点、优势、局限
- 如果 metrics.csv 不存在，注明"实验结果待补充"

**Prediction and Conclusions**
- 主要发现：哪个模型表现最好、在什么条件下
- 洞察：为什么 ARIMA 在这类数据上有优势（或其他发现）
- 挑战：遇到的具体技术问题（列举 1-3 个有价值的）
- 局限性：数据量、模型假设、评估方法的潜在不足
- 建议：如果继续做可以改进什么（从 TECHNICAL.md 的"已知限制"部分提取）
- 结论：一段总结性陈述

---

## 第三步：写入文件

用 Write 工具将完整报告写入 `report.md`（项目根目录），全量覆盖。

写完后告知用户：
- 报告已更新
- 封面页成员信息需手动填写
- 哪些章节因数据不足标注了"待补充"（如有）
