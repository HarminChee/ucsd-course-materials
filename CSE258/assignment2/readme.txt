F1 赛事预测项目阶段性总结（截至 2025-11-26）
一、数据来源与文件结构
原始数据来源
Kaggle 数据集：
(1) Formula 1 World Championship (1950–2024) by Rohan Rao
提供多张核心表：races, results, drivers, constructors, driver_standings 等。
(2) Formula 1 weather info (1950–2024)
当前只下载并解压为 weather_features_v4.csv，暂未在特征工程中使用。

本地文件夹结构（根目录）
根目录：C:\Users\xxx\Desktop\f1_project\
原始数据：
C:\Users\xxx\Desktop\f1_project\data_raw\f1_core
包含：
circuits.csv
constructors.csv
constructor_results.csv
constructor_standings.csv
driver_standings.csv
drivers.csv
lap_times.csv
pit_stops.csv
qualifying.csv
races.csv
results.csv
seasons.csv
sprint_results.csv
status.csv
C:\Users\xxx\Desktop\f1_project\data_raw\f1_weather
包含：
weather_features_v4.csv

*****处理后数据：
C:\Users\harmi\Desktop\f1_project\data_processed
目前生成的两个核心数据集：
(1) task1_race_driver_podium.csv
用于“单场比赛领奖台（三甲）预测”的样本表。
(2) task2_driver_season_points_midround.csv
用于“赛季最终积分预测”的样本表。

*****代码：
C:\Users\harmi\Desktop\f1_project\src
(1) build_f1_datasets.py
负责从原始 CSV 构建 Task1 / Task2 的特征数据集。
(2) train_task1_baselines.py
负责在 Task1 数据上训练与评估基线分类模型。
(3) train_task2_baselines.py
负责在 Task2 数据上训练与评估基线回归模型。



二、数据处理与建模流程（已经完成的工作）

年份过滤与基本整理

在 build_f1_datasets.py 中：

从 races.csv 中筛选 year ≥ 2000 的“近代 F1 赛季”，共 479 场比赛。

将 results.csv 与 driver_standings.csv 中只保留这些 raceId 对应的记录：

results_2000plus：10079 条 (单场比赛-车手级别结果)

driver_standings_2000plus：10552 条 (比赛后车手积分/排名记录)

Task 1：单场比赛领奖台（三甲）预测数据构建

样本粒度：每一行代表“某一年某一场比赛中的某一位车手”
key: (raceId, driverId)

使用的原始表：

races.csv：提供 year, round, circuitId 等比赛信息；

results.csv：提供每场比赛中每位车手的 grid（发车位）、points（得分）、positionOrder（完赛名次）等；

drivers.csv：提供车手基本信息；

constructors.csv：提供车队基本信息；

driver_standings.csv：提供赛后车手积分和排名，用于构造赛前积分特征。

主要特征：
(1) 当场静态特征：
- year, round, circuitId
- grid（发车位）
(2) 赛前积分与排名：
- 从 driver_standings 中按 (year, driverId, round) 排序，
将每场比赛后的 points, position 向后 shift 一场，
得到 prerace_points（赛前累计积分）和 prerace_rank（赛前排名）。
(3) 历史 Rolling 特征（最近 3 场）：
- prev1_finish：上一场比赛名次
- prev3_avg_finish：最近 3 场平均完赛名次
- prev3_points：最近 3 场累计得分
通过 groupby(driverId) + rolling(window=3) + shift(1) 构造，缺失值用中位数填充。

标签：

finish_pos：最终完赛名次（来自 positionOrder）

is_podium：是否登上领奖台（三甲），定义为 1 ≤ finish_pos ≤ 3 的二分类标签。

最终生成文件：

data_processed\task1_race_driver_podium.csv

Task 2：赛季最终积分预测数据构建

样本粒度：每一行代表“某一年某一位车手的赛季表现”
key: (year, driverId)

使用的原始表：

races.csv：提供 year, round，用于计算每赛季总场次及 mid_round（赛程中点）；

results.csv：提供每场得分与名次，用来构造赛季前半程统计特征；

driver_standings.csv：提供每场后车手累计积分与排名，用来抽取赛季最终积分和排名。

标签（赛季最终表现）：

对每个 (year, driverId)，按 round 排序，取赛季最后一场记录：

final_points：赛季最终总积分

final_rank：赛季最终积分排名

特征构造：
(1) 赛季中点 mid_round：
- 对每个 year 统计 max_round，然后 mid_round = floor(max_round / 2)。
(2) 前半赛季（round ≤ mid_round）聚合特征：
- mid_points：前半赛季总积分
- mid_avg_finish：前半赛季平均完赛名次
- mid_races：前半赛季参加的比赛场数
- mid_wins：前半赛季获胜场次
- mid_podiums：前半赛季登上领奖台场次
(3) 上一赛季历史信息：
- 取 ds_final（上一段求出的每个赛季的 final_points, final_rank），
将 year + 1 后，与当前 (year, driverId) merge：
- prev_final_points：上一赛季最终积分（无上一赛季则填 0）
- prev_final_rank：上一赛季最终排名（无上一赛季则用最大排名填补）

最终生成文件：

data_processed\task2_driver_season_points_midround.csv

训练/验证/测试划分策略（两项任务共用）

按年份做时间切分，避免未来信息泄露：

训练集（train）：2000–2014 年

验证集（val）：2015–2018 年

测试集（test）：2019–2023 年

这样保证测试集完全在时间上“晚于”训练集，更贴近真实预测场景。

Task 1：模型与结果

评价目标：

二分类指标（对“车手是否登上领奖台”）：ROC-AUC, F1；

排名型指标（按概率排序）：每场比赛的 Recall@3、Top-1 冠军预测准确率。

(1) Baseline：基于发车位的简单规则
- 做法：对每场比赛，直接选择发车位最靠前的 3 个车手作为预测“领奖台”。
- 测试集结果（2019–2023）：
- Recall@3（三甲命中率）：0.5737
- Top-1 冠军准确率：0.3365
- 解读：
发车位本身已经是一个非常强的先验信息，只看 grid 就能在大约 57% 的真实领奖台车手中命中，并且 33% 的比赛可以预测对冠军。

(2) Logistic Regression（带 StandardScaler，class_weight=balanced）
- 特征：grid, year, round, prerace_points, prerace_rank, prev1_finish, prev3_avg_finish, prev3_points。
- 测试集结果：
- Global ROC-AUC：0.9235
- Global F1-score：0.6374
- Recall@3：0.6506
- Top-1 冠军准确率：0.5962
- 解读：
- ROC-AUC 接近 0.92，说明模型在区分“上领奖台”和“未上领奖台”方面整体表现很强；
- F1 分数 0.64，兼顾了 precision 和 recall；
- Recall@3 从 baseline 的 0.57 提升到 0.65，说明在需要预测每场前三名时，模型比简单“按发车位排序”多命中了一部分真实领奖台车手；
- Top-1 冠军预测准确率从 0.34 提升到 0.60，非常显著，说明模型结合了赛前积分与近期表现等信息后，对冠军预测有较大提升。

(3) Random Forest Classifier
- 参数：n_estimators=300, class_weight="balanced_subsample" 等。
- 测试集结果：
- Global ROC-AUC：0.9211（与 LogReg 接近）
- Global F1-score：0.5292（明显低于 LogReg）
- Recall@3：0.6410（略优于 baseline，但略低于 LogReg）
- Top-1 冠军准确率：0.3654（只比 baseline 略好，远低于 LogReg）
- 解读：
- 虽然整体 AUC 也很高，但在 F1 和 Top-1 上的表现不如 Logistic Regression；
- 可能原因：特征数量相对有限且多为“弱线性特征”（积分、名次等），LogReg 更容易学到稳定的线性决策边界；而 RF 对超参数较敏感，且在相对小样本（几千）+ 强相关特征场景下容易过拟合或学到相对“粗糙”的排名。

小结（Task1）：

我们已经有了三个可对比的 baseline：
(1) Grid-based rule（强基线，靠发车位排序）；
(2) Logistic Regression（当前效果最佳的简单模型）；
(3) Random Forest（非线性树模型，对未来复杂模型有参考意义）。




从结果看：

Logistic Regression 在所有关键指标上都显著优于 grid baseline；

说明我们构造的赛前积分、滚动历史表现等特征，对领奖台预测有实质贡献；

也说明这个任务是“可学的”，数据质量足以支持更复杂模型。

Task 2：赛季最终积分预测模型与结果

评价目标：

回归指标：MAE, RMSE, R²；

排序指标：Spearman rank correlation（预测积分与真实积分的排序相关性）。

(1) Baseline：mid_points（只看前半赛季积分）
- 做法：直接用 mid_points 作为 final_points 预测。
- 测试集结果（2019–2023）：
- MAE：55.21
- RMSE：84.06
- R²：0.5151
- Spearman：0.9723
- 解读：
- R² ≈ 0.51，说明仅凭前半赛季积分就能解释约 50% 的赛季最终积分方差；
- Spearman ≈ 0.97，说明前半赛季积分对“排序”非常有参考价值：只看 mid_points 已经几乎能排出谁是高分车手，谁是低分车手（但数值误差较大）。

(2) Linear Regression（StandardScaler + LinearRegression）
- 特征：mid_points, mid_avg_finish, mid_races, mid_wins, mid_podiums, prev_final_points, prev_final_rank。
- 测试集结果：
- MAE：15.93
- RMSE：23.64
- R²：0.9617
- Spearman：0.9679
- 解读：
- 与 baseline 相比，MAE 从 55 降到 16 左右，RMSE 从 84 降到 24 左右，误差大幅降低；
- R² ≈ 0.96，说明线性回归能解释约 96% 的赛季最终积分方差，拟合效果非常好；
- Spearman 仍然在 0.96–0.97 之间，说明排序表现依然很强，且和 baseline 排序相差不大（本身 baseline 排序就已经很强）。
- 这表明：如果已知“前半赛季的积分/表现 + 上一赛季的积分/排名”，线性模型已经可以非常准确地估计赛季最终积分。

(3) RandomForestRegressor
- 参数：n_estimators=400 等。
- 测试集结果：
- MAE：19.94
- RMSE：32.55
- R²：0.9273
- Spearman：0.9632
- 解读：
- RF 的表现明显优于 baseline，但略逊于 Linear Regression；
- 说明在当前特征维度和样本大小下，线性关系已经足够解释大部分信息，复杂非线性模型带来的边际收益有限。

小结（Task2）：

赛季最终积分预测在当前特征下属于“相对容易”的任务；

线性模型的 R² 达到 0.96，说明我们构造的“前半赛季表现 + 上赛季最终积分”特征，几乎可以重构后半赛季的积分；

这为后续更复杂的 time-series 模型（LSTM / Transformer 等）提供了一个非常强的 baseline：任何复杂模型至少要在 MAE / RMSE 上接近或优于当前线性回归。

三、目前工作的意义与项目下一步方向

已经完成的核心工作意义

完成了从原始 F1 比赛数据，到“任务可用数据集（Task1/Task2）”的全流程自动化脚本：

具备可复现性（build_f1_datasets.py 可一键从原始 CSV 生成 processed CSV）；

特征设计考虑了现实场景：赛前积分、近期表现、赛季中点之前的统计信息等；

在两个任务上建立了多种 baseline：

Task1：强规则（grid）、Logistic Regression、Random Forest；

Task2：mid_points baseline、线性回归、Random Forest 回归；

初步结论：

领奖台预测任务具有一定难度，但我们的模型明显优于强规则基线；

赛季积分预测在当前特征下非常可学，线性模型已经接近“高拟合度”。

下一步可拓展方向（可以写进项目 proposal）

特征层面：

引入 weather_features_v4.csv（温度、降水、风速等）；

加入赛道特征（街道赛/传统赛道、弯道数量、赛道长度等）；

加入车队层面的历史表现（车队平均积分、车队技术规则变更前后差异等）。

模型层面：

对 Task1 采用学习排序模型（LambdaMART、XGBoost Rank）或神经网络；

对 Task2 与 Task1 都可以尝试 RNN/LSTM/Temporal CNN 处理整赛季时间序列；

推荐系统视角：

将“车手在某场比赛的表现”看作对“比赛领奖台名额”的“曝光-点击”问题，
探索用 CTR 预估/学习排序框架来建模（NDCG@3, MAP@3 等指标）。