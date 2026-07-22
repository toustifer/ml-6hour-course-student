# 机器学习 6 小时课 · 学生材料

咖啡馆营收案例：Python 基础 → Pandas → 决策树 / 随机森林。

面向零基础同学。本仓库**只含学生用教案与数据**，不含教师答案与口播稿。

## 目录

```text
教学课程/
  day1/   Python 与读表入门
  day2/   Pandas 与简单特征
  day3/   机器学习概念、决策树、随机森林
项目内容/  案例相关补充（可选）
```

每天建议从对应目录下的 `dayN_6h_00_...` 或说明文件按顺序打开 `.ipynb`。

## 环境建议

- Anaconda / Miniconda
- 课程常用环境名：`day1_ml`（Python 3.11 左右）
- 主要库：`pandas`、`matplotlib`、`scikit-learn`（Day 3）

```bash
conda activate day1_ml
# 若缺包：
# conda install pandas matplotlib scikit-learn -y
```

在 VS Code / Jupyter 中把工作目录切到当天的 `教学课程/dayN`，内核选 `day1_ml`，再从上到下运行 notebook。

## 使用说明

1. 克隆或下载本仓库  
2. 打开对应天的文件夹  
3. 按 notebook 编号顺序学习  
4. 数据 CSV 与 notebook 放在同一天目录内，一般无需改路径  

## 仓库说明

- 公开学生向：`ml-6hour-course-student`（本仓库）  
- 教师完整材料（含答案、口播、节奏）在教师本机 / 私有仓库，不在此发布  

有问题优先看当堂 notebook 里的说明；环境报错先检查是否装对环境和内核。
