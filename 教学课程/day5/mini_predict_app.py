# -*- coding: utf-8 -*-
"""
Day 5 迷你版：和 Day 3/4 同一套特征的「训练 + 预测」网页。
课堂跟练可先跑这个，完整 system/ 下午再演示。

用法（在 day5 目录）：
  conda activate day1_ml
  python mini_predict_app.py
  浏览器打开 http://127.0.0.1:5001
"""
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "day5_cafe_sales.csv"

FEATURE_COLS = [
    "price",
    "promotion",
    "is_weekend",
    "temp",
    "quality",
    "competitors",
    "holiday",
    "location",
    "weather_score",
]
WEATHER_MAP = {"晴": 1.0, "多云": 0.8, "阴": 0.6, "小雨": 0.4, "大雨": 0.3}

CACHE = {}


def load_xy():
    df = pd.read_csv(CSV_PATH)
    df = df.copy()
    df["weather_score"] = df["weather_label"].map(WEATHER_MAP)
    X = df[FEATURE_COLS]
    y = df["sales"]
    return X, y


def ensure_model():
    if "model" in CACHE:
        return CACHE
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    CACHE["model"] = rf
    CACHE["features"] = FEATURE_COLS
    CACHE["r2"] = round(float(r2_score(y_test, pred)), 3)
    CACHE["mae"] = round(float(mean_absolute_error(y_test, pred)), 1)
    CACHE["importance"] = sorted(
        [
            {"feature": c, "importance": round(float(v), 4)}
            for c, v in zip(FEATURE_COLS, rf.feature_importances_)
        ],
        key=lambda d: -d["importance"],
    )
    return CACHE


PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>迷你营收预测</title>
  <style>
    body{font-family:"Microsoft YaHei",sans-serif;background:#f8f4f0;margin:0;padding:24px;color:#2c2c2c}
    h1{color:#6f4e37}
    .card{background:#fff;border-radius:12px;padding:16px;max-width:720px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:16px}
    label{display:inline-block;width:110px;font-size:13px;color:#666}
    input{padding:6px 8px;margin:4px 0;border:1px solid #ddd;border-radius:6px;width:140px}
    button{background:#6f4e37;color:#fff;border:none;border-radius:8px;padding:8px 16px;cursor:pointer;margin-right:8px}
    button:hover{background:#4a3528}
    #out{margin-top:12px;padding:12px;background:#f0ebe6;border-radius:8px;min-height:40px}
    code{background:#f0ebe6;padding:2px 6px;border-radius:4px}
  </style>
</head>
<body>
  <h1>迷你营收预测（Day 5）</h1>
  <div class="card">
    <p>特征与 Day 3/4 一致。先点「训练」，再填表「预测」。</p>
    <button onclick="doTrain()">训练随机森林</button>
    <div id="trainInfo"></div>
  </div>
  <div class="card">
    <div><label>price</label><input id="price" value="25"></div>
    <div><label>promotion</label><input id="promotion" value="1"></div>
    <div><label>is_weekend</label><input id="is_weekend" value="1"></div>
    <div><label>temp</label><input id="temp" value="22"></div>
    <div><label>quality</label><input id="quality" value="8"></div>
    <div><label>competitors</label><input id="competitors" value="2"></div>
    <div><label>holiday</label><input id="holiday" value="0"></div>
    <div><label>location</label><input id="location" value="7"></div>
    <div><label>weather_score</label><input id="weather_score" value="1.0">（晴≈1.0 大雨≈0.3）</div>
    <p style="margin-top:12px"><button onclick="doPredict()">预测营收</button></p>
    <div id="out">预测结果会显示在这里</div>
  </div>
  <script>
  async function doTrain(){
    const r = await fetch('/api/train', {method:'POST'});
    const j = await r.json();
    document.getElementById('trainInfo').innerHTML =
      j.status==='success'
        ? `<p>测试 R² = <b>${j.r2}</b>，MAE = <b>${j.mae}</b></p>
           <p>Top 特征：${j.importance.slice(0,3).map(x=>x.feature).join('、')}</p>`
        : `<p style="color:red">${j.message||'失败'}</p>`;
  }
  async function doPredict(){
    const body = {};
    ['price','promotion','is_weekend','temp','quality','competitors','holiday','location','weather_score']
      .forEach(k => body[k] = parseFloat(document.getElementById(k).value));
    const r = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const j = await r.json();
    document.getElementById('out').textContent =
      j.status==='success' ? ('预测营收 ≈ ' + j.prediction + ' 元') : (j.message||'失败');
  }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        info = ensure_model()
        return jsonify(
            {
                "status": "success",
                "r2": info["r2"],
                "mae": info["mae"],
                "importance": info["importance"],
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "model" not in CACHE:
        ensure_model()
    data = request.get_json(force=True, silent=True) or {}
    try:
        row = {c: float(data.get(c, 0)) for c in FEATURE_COLS}
        x = pd.DataFrame([row], columns=FEATURE_COLS)
        pred = float(CACHE["model"].predict(x)[0])
        return jsonify({"status": "success", "prediction": round(pred, 1)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/hello")
def api_hello():
    return jsonify({"msg": "hello from mini app", "ok": True})


if __name__ == "__main__":
    print("迷你预测：http://127.0.0.1:5001")
    print("CSV:", CSV_PATH, "exists=", CSV_PATH.exists())
    app.run(debug=True, host="127.0.0.1", port=5001, use_reloader=False)
