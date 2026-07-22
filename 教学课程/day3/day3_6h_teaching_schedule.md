# Day 3 授课节奏安排

这份文件给老师自己看，不放进学生教案内容里。

> **说明：** 本文件是老师侧时长安排，学生主看 `day3_6h_0*.ipynb`。  
> 教师口播/答案/可抄代码等仅保存在本地 `教师资料/`（不进公开学生仓库）。

## 正确主线顺序（非常重要）

```
模块 0 复习与环境自检
   ↓
模块 1 机器学习是什么（概念为主）
   ↓
模块 2 决策树：像一串 if
   ↓
模块 3 从一棵树到一片森林
   ↓
模块 4 用 sklearn 训练随机森林
   ↓
模块 5 特征重要性：谁更重要
   ↓
模块 6 预测与简单评估（R² 入门）
```

- Day 3 的重点不是推公式，而是**能训出一个随机森林，并读懂特征重要性**
- Day 3 默认学生已经学过：读 CSV、选列过滤、groupby、简单新列、train/test 概念
- `07` 是课堂练习册，`08` 是课后练习册，仍然不是新课主体
- 超参网格搜索、多算法对比留给 Day 4；今天只轻触 `n_estimators` / `max_depth`

## 总原则

Day 3 的目标是让学生从“会操作表”走到“会训一个模型”：

1. 能用人话解释：机器学习在学什么
2. 知道决策树是一串条件判断
3. 知道随机森林 = 很多树投票/平均
4. 会用 sklearn 训练 `RandomForestRegressor`
5. 会看 `feature_importances_`
6. 会做一次预测，并知道 R² 大致表示“解释了多少波动”

如果课堂节奏变慢，优先保证：

- 准备 X / y + train_test_split
- 训练一棵浅决策树（`export_text` + 建议 `plot_tree`，测试集 `predict` 对照）
- 训练随机森林
- 输出特征重要性
- 一次 `predict` + 看 R²

树图美化、系统超参网格、残差分析可以讲浅一点。

## 建议安排（6 小时）

| 阶段 | 内容 | 建议时长 | 说明 |
| ---- | ---- | ---- | ---- |
| 0 | 模块 0：复习与环境自检 | 20-30 分钟 | 确认 sklearn、csv、内核 |
| 1 | 模块 1：机器学习概念 | 40-50 分钟 | 监督/无监督一句、回归/分类、过拟合/欠拟合，少代码 |
| 2 | 模块 2：决策树 | 60-75 分钟 | if 串 + 浅树 + export_text + plot_tree + 测试 predict |
| 3 | 模块 3：从树到森林 | 40-50 分钟 | Bagging 比喻，少公式 |
| 4 | 模块 4：训练随机森林 | 50-60 分钟 | fit / predict / 与树对比 |
| 5 | 模块 5：特征重要性 | 40-50 分钟 | 读 importance，说业务话 |
| 6 | 模块 6：预测与 R² 入门 | 40-50 分钟 | 单样本预测 + score |
| 7 | 练习册 `07` | 穿插进行 | 跟着模块 2-6 做 |
| 8 | 课后练习 `08` | 5-10 分钟布置 | 回家独立完成一遍 |

## 文件角色对照

| 文件 | 角色 |
| ---- | ---- |
| `day3_6h_00_review_and_setup.ipynb` | 复习与环境自检 |
| `day3_6h_01_ml_concepts.ipynb` | 机器学习概念 |
| `day3_6h_02_decision_tree.ipynb` | 决策树 |
| `day3_6h_03_from_tree_to_forest.ipynb` | 从树到森林 |
| `day3_6h_04_train_random_forest.ipynb` | 训练随机森林 |
| `day3_6h_05_feature_importance.ipynb` | 特征重要性 |
| `day3_6h_06_predict_and_score.ipynb` | 预测与简单评估 |
| `day3_6h_07_classroom_workbook.ipynb` | 课堂练习册 |
| `day3_6h_08_homework_workbook.ipynb` | 课后练习册 |

## 最低合格线

学生当天至少能独立完成：

1. 准备 `X`（特征）和 `y`（营收）
2. `train_test_split`
3. 训练 `RandomForestRegressor` 并 `fit`
4. 打印 `feature_importances_`（或排序后的重要性表）
5. 对测试集 `predict`，并看到一个 R² 分数
