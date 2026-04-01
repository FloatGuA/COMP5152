# COMP5152 — Progress

## 状态快照
| 项目       | 值           |
|------------|--------------|
| 整体状态   | 进行中       |
| 最后更新   | 2026-04-01   |

## 已完成
- 将实验资产 CEZ 替换为 BND（CEZ 在 Prophet/LSTM 上出现异常 MAPE，疑似价格尺度问题）
- 跑通首次正式实验（6只资产×4模型）：ARIMA 均值 MAPE 1.40% 最优，LR 2.79%，Prophet 31.62%，LSTM 114.25%（受 CEZ 拉高）
- 新增 plot_results.py 可视化脚本：MAPE bar/heatmap/grouped、训练时间 bar、LSTM loss 曲线，共 5 张图
- 新增模型缓存模块 src/model_cache.py：四模型训练结果持久化到 models/，cache key 为 train_end + val_end 日期
- 新增 tqdm 进度条：run_experiment.py 总进度、LSTM epoch 级别（train/val loss + patience）、ARIMA/Prophet rolling 步进度
- 修复 LR target 设计缺陷：target 改为 next-day Close（shift(-1)），MAPE 从虚假 0.5% 恢复正常
- 修复 ARIMA：改为 rolling one-step 预测，每 20 步全量 refit
- 修复 Prophet：改为 rolling one-step 预测，每 30 步全量 refit
- 搭建完整建模框架（src/ 下 5 个模型模块 + evaluator + data_loader + run_experiment.py）
- 随机抽取 6 只 A 级资产（CLI/ALCO/ACCO/DWM/CHII/CEZ），跑通 6×4 首次实验
- 发现并定位三个评估问题：LR target 设计缺陷、ARIMA/Prophet fixed-origin 评估失真、LSTM 已是 rolling 评估
- 确定修复方案并写入 todolist（4 个任务）
- 确定实验设计：7:2:1 时序切分，MAPE 为主指标，1/n 等权平均
- 讨论项目可行性，确定四模型对比方案（Linear Regression / ARIMA / Prophet / LSTM）
- 确定数据分级标准（A/B/C/D 四级，按年限+行数双维度筛选，附数据密度计算）
- 编写 filter_datasets.py，实现 CSV 自动分类、输出到 output/{grade}/{category}/ 目录
- 对前 1/5 数据（etfs 433个 + stocks 1177个，共 1609 个文件）跑通脚本，结果 A:809 B:306 C:135 D:359

## 进行中 / 待处理
- 用 BND 替换 CEZ 后重跑实验，收集正确的全量指标
- [Task 4] 新增统一评估模块：data_group（1yr/5yr/full）维度，composite score（0.5×MAPE+0.3×RMSE+0.2×MAE，min-max norm），heatmap/bar chart
- 最终报告撰写

## 变更记录
- 2026-04-01：项目初始化，完成数据筛选脚本 filter_datasets.py，测试通过前 1/5 数据
- 2026-04-01：搭建建模框架，完成首次 6×4 实验，定位评估问题，制定修复 todolist
- 2026-04-01：修复三个评估问题（LR/ARIMA/Prophet），新增缓存/进度条/可视化，完成首次正式实验，CEZ 替换为 BND
