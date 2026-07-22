# Day 5 演示系统（完整版）

从项目成品复制的咖啡馆营收 Web 系统，**下午整合课**主要跑这一套。

## 目录

```text
system/
  app.py              # Flask 后端：读 CSV、训 RF、API
  cafe_sales.csv      # 数据（与 day5_cafe_sales 同套）
  templates/
    index.html        # 前端单页（首页/数据/可视化/训练/预测）
```

## 启动（在 system 目录下）

```text
conda activate day1_ml
# 若缺 flask：
conda install flask -y
# 或：pip install flask

cd 教学课程/day5/system
python app.py
```

浏览器打开：`http://127.0.0.1:5000`

## 和前四天的对应

| 系统里 | 你学过的 |
| --- | --- |
| `load_and_clean` | Day 2 读表、清洗、天气分 |
| `train_model` + RF | Day 3–4 训森林、R²、重要性 |
| `/api/train` | 把 fit/score 放进接口 |
| `/api/predict` | 把 `predict` 放进接口 |
| 前端表单 | 人在网页上填特征 |

## 注意

- 先点「训练」，再点「预测」（内存里要有模型）。
- 关闭终端或 Ctrl+C 会停掉服务。
- 不要改端口冲突时，可看 `app.py` 最后 `port=5000`。
