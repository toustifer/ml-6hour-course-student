#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
咖啡馆营收数据分析系统 — Flask + 随机森林
数据读取 → 随机森林建模 → 营收预测 → 可视化看板
"""

import os, json, random, uuid
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)
app.secret_key = 'cafe_ai_2026'

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, 'cafe_sales.csv')
MODEL_PATH = os.path.join(BASE, 'model_cache.json')

# ===================== 数据加载与清洗 =====================
def load_and_clean(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    # 标准化列名映射
    col_map = {}
    col_rules = {
        '日期': 'date', 'day': 'day',
        '客流量': 'traffic', '单品售价': 'unit_price', '单价': 'unit_price',
        'price': 'unit_price',
        '促销': 'promotion', '促销活动': 'promotion',
        '天气': 'weather', '天气评分': 'weather_score',
        '天气标签': 'weather_label', 'weather_label': 'weather_label',
        '节假日': 'holiday', '节日': 'holiday',
        '原材料成本': 'material_cost', '成本': 'material_cost',
        '门店营收': 'revenue', '营收': 'revenue', '销售额': 'revenue', 'sales': 'revenue',
        '客流量': 'traffic', 'traffic': 'traffic',
        '气温': 'temp', '温度': 'temp', 'temp': 'temp',
        '品质': 'quality', 'quality': 'quality',
        '位置': 'location', 'location': 'location',
        '竞争对手': 'competitors', 'competitors': 'competitors',
        '周末': 'is_weekend', 'is_weekend': 'is_weekend',
    }
    for c in df.columns:
        for k, v in col_rules.items():
            if k in c:
                col_map[c] = v
                break
    df.rename(columns=col_map, inplace=True)

    # 缺失值处理 — 只处理数值型列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # 生成天气评分用于建模（0=晴→1.0, 4=大雨→0.3）
    if 'weather' in df.columns and 'weather_score' not in df.columns:
        w_map = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.3}
        df['weather_score'] = df['weather'].map(w_map).fillna(0.6)
    if 'temp' not in df.columns and 'temp' in df.columns:
        pass
    # 确保unit_price存在
    if 'unit_price' not in df.columns and 'price' in df.columns:
        df['unit_price'] = df['price']
    # 确保traffic存在—用随机模拟（原始数据没有客流量）
    if 'traffic' not in df.columns:
        df['traffic'] = (df['revenue'] / df['unit_price']).round(0).astype(int) if 'unit_price' in df.columns else np.random.randint(50, 200, len(df))

    # 营收异常过滤（超过3倍标准差视为异常）
    if 'revenue' in df.columns:
        mean_r, std_r = df['revenue'].mean(), df['revenue'].std()
        df = df[df['revenue'].between(mean_r - 3 * std_r, mean_r + 3 * std_r)]

    return df


# ===================== 随机森林建模 =====================
FEATURE_COLS = ['traffic', 'unit_price', 'promotion', 'holiday',
                'quality', 'temp', 'competitors', 'is_weekend', 'location',
                'weather_score']
TARGET = 'revenue'

MODEL_CACHE = {}


def train_model(df):
    """训练随机森林，返回模型和评估指标"""
    # 构建特征 & 确保数值类型
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for c in available:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X.fillna(X.median(), inplace=True)

    y = pd.to_numeric(df[TARGET], errors='coerce')
    y.fillna(y.median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = round(r2_score(y_test, y_pred), 4)
    mse = round(mean_squared_error(y_test, y_pred), 2)

    importance = sorted(
        [{'feature': col, 'importance': round(float(imp), 4)}
         for col, imp in zip(available, rf.feature_importances_)],
        key=lambda x: x['importance'], reverse=True
    )

    MODEL_CACHE['model'] = rf
    MODEL_CACHE['features'] = available
    MODEL_CACHE['r2'] = r2
    MODEL_CACHE['mse'] = mse
    MODEL_CACHE['importance'] = importance
    MODEL_CACHE['y_true'] = [round(float(v)) for v in y_test.tolist()]
    MODEL_CACHE['y_pred'] = [round(float(v)) for v in y_pred.tolist()]

    return MODEL_CACHE


# ===================== Flask 路由 =====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/data/summary')
def api_data_summary():
    df = load_and_clean(DATA_PATH)
    return jsonify({
        'total_days': len(df),
        'avg_revenue': round(float(df['revenue'].mean()), 2) if 'revenue' in df else 0,
        'max_revenue': round(float(df['revenue'].max()), 2) if 'revenue' in df else 0,
        'avg_traffic': round(float(df['traffic'].mean()), 1) if 'traffic' in df else 0,
        'max_traffic': int(df['traffic'].max()) if 'traffic' in df else 0,
        'avg_cost': round(float(df['material_cost'].mean()), 2) if 'material_cost' in df else 0,
        'columns': list(df.columns),
        'rows': len(df),
    })


@app.route('/api/data/table')
def api_data_table():
    df = load_and_clean(DATA_PATH)
    page = request.args.get('page', 1, type=int)
    size = request.args.get('size', 20, type=int)
    sort = request.args.get('sort', '')
    order = request.args.get('order', 'asc')

    if sort and sort in df.columns:
        df = df.sort_values(sort, ascending=(order == 'asc'))

    total = len(df)
    start = (page - 1) * size
    end = min(start + size, total)

    rows = []
    for _, r in df.iloc[start:end].iterrows():
        rows.append({k: (v if not pd.isna(v) else None) for k, v in r.items()})

    return jsonify({'rows': rows, 'total': total, 'page': page, 'size': size})


@app.route('/api/data/overview')
def api_data_overview():
    df = load_and_clean(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    result = {
        'daily_revenue': [],
        'daily_traffic': [],
    }
    if 'date' in df.columns:
        date_col = df['date'].astype(str)
        if 'revenue' in df:
            for i, (d, v) in enumerate(zip(date_col, df['revenue'])):
                result['daily_revenue'].append({'day': i + 1, 'date': str(d)[:10], 'value': round(float(v), 2)})
        if 'traffic' in df:
            for i, (d, v) in enumerate(zip(date_col, df['traffic'])):
                result['daily_traffic'].append({'day': i + 1, 'date': str(d)[:10], 'value': int(v)})
    else:
        if 'revenue' in df:
            for i, v in enumerate(df['revenue']):
                result['daily_revenue'].append({'day': i + 1, 'date': f'Day {i + 1}', 'value': round(float(v), 2)})
        if 'traffic' in df:
            for i, v in enumerate(df['traffic']):
                result['daily_traffic'].append({'day': i + 1, 'date': f'Day {i + 1}', 'value': int(v)})
    return jsonify(result)


@app.route('/api/data/category')
def api_data_category():
    df = load_and_clean(DATA_PATH)
    if 'category' not in df.columns or 'revenue' not in df.columns:
        return jsonify({'data': []})

    grp = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
    return jsonify({
        'data': [{'name': k, 'value': round(float(v))} for k, v in grp.items()]
    })


@app.route('/api/data/scatter')
def api_data_scatter():
    df = load_and_clean(DATA_PATH)
    if 'traffic' not in df.columns or 'revenue' not in df.columns:
        return jsonify({'data': []})
    data = []
    for _, r in df.iterrows():
        data.append({
            'traffic': int(r['traffic']),
            'revenue': round(float(r['revenue']), 2),
            'promotion': int(r['promotion']) if 'promotion' in r else 0,
            'holiday': int(r['holiday']) if 'holiday' in r else 0,
        })
    return jsonify({'data': data})


@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        df = load_and_clean(DATA_PATH)
        result = train_model(df)
        return jsonify({
            'status': 'success',
            'r2': result['r2'],
            'mse': result['mse'],
            'importance': result['importance'],
            'y_true': result['y_true'][:100],
            'y_pred': result['y_pred'][:100],
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'model' not in MODEL_CACHE:
        return jsonify({'status': 'error', 'message': '模型未训练，请先训练'})

    data = request.get_json() or {}
    features = MODEL_CACHE['features']
    rf = MODEL_CACHE['model']

    try:
        x = []
        for col in features:
            val = data.get(col, 0)
            x.append(float(val))
        x_arr = np.array(x).reshape(1, -1)
        pred = float(rf.predict(x_arr)[0])
        return jsonify({'status': 'success', 'prediction': round(pred, 2)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    print("☕ 咖啡馆营收数据分析系统")
    print("=" * 40)
    if os.path.exists(DATA_PATH):
        df = load_and_clean(DATA_PATH)
        print(f"[OK] 数据集: {len(df)} 行")
        print(f"[OK] 列: {list(df.columns)}")
        train_model(df)
        print(f"[OK] 模型预热完成 | R² = {MODEL_CACHE.get('r2', 'N/A')}")
    else:
        print(f"[!] 未找到数据文件: {DATA_PATH}")
        print(f"    请将咖啡馆数据集命名为 cafe_data.csv 放在项目目录")
    print(f"\n  访问地址: http://127.0.0.1:5000")
    print("=" * 40)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
