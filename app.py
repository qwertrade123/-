# -*- coding: utf-8 -*-
"""
程式交易回測 Streamlit APP
策略：(一)移動平均 (二)RSI順勢 (三)RSI逆勢 (四)布林通道 (五)MACD (六)KDJ (七)自定義
功能：K棒圖、策略回測、參數最佳化（Sharpe+Calmar）、AI 績效評估
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os, sys, warnings, itertools, io
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import plotly.graph_objects as go

from talib.abstract import SMA, RSI, BBANDS, MACD, STOCH

# 載入同目錄的自定義模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from order_Lo8 import Record

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 資料前處理
# ─────────────────────────────────────────────────────────────────────────────
def build_kbar_dic(df: pd.DataFrame, cycle_minutes: int) -> dict:
    """將 DataFrame 正規化並聚合成指定週期的 KBar 字典。"""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # 欄位別名對應
    rename = {}
    for c in df.columns:
        if c in ('ts', 'datetime', 'date', 'timestamp', 'index'):
            rename[c] = 'time'
        elif c == 'vol':
            rename[c] = 'volume'
        elif c == 'amt':
            rename[c] = 'amount'
    df = df.rename(columns=rename)

    # 確保有 time 欄位
    if 'time' not in df.columns:
        df['time'] = df.index
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df.set_index('time')

    for col in ('open', 'high', 'low', 'close', 'volume'):
        if col not in df.columns:
            if col == 'volume':
                df[col] = 1.0
            else:
                raise ValueError(f"資料缺少必要欄位: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 聚合到指定週期（pandas resample）
    if cycle_minutes > 1:
        rule = f'{cycle_minutes}min'
        df = df.resample(rule).agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum',
        }).dropna(subset=['open', 'close'])

    df = df.reset_index()
    kd = {
        'time':    df['time'].values,
        'open':    df['open'].values.astype(np.float64),
        'high':    df['high'].values.astype(np.float64),
        'low':     df['low'].values.astype(np.float64),
        'close':   df['close'].values.astype(np.float64),
        'volume':  df['volume'].values.astype(np.float64),
        'product': np.repeat('stock', len(df)),
    }
    return kd


# ─────────────────────────────────────────────────────────────────────────────
# 七個策略函數
# ─────────────────────────────────────────────────────────────────────────────
def _cover_all_long(rec, kd, n):
    rec.Cover('Sell', kd['product'][n+1], kd['time'][n+1], kd['open'][n+1], rec.GetOpenInterest())

def _cover_all_short(rec, kd, n):
    rec.Cover('Buy', kd['product'][n+1], kd['time'][n+1], kd['open'][n+1], -rec.GetOpenInterest())


def run_MA(kd, long_period=10, short_period=2, stop_loss=10.0, order_qty=1):
    """(一) 移動平均線策略：短均線上穿長均線買入，下穿賣出，移動停損。"""
    rec = Record()
    kd2 = dict(kd)
    kd2['MA_long']  = SMA(kd2, timeperiod=int(long_period))
    kd2['MA_short'] = SMA(kd2, timeperiod=int(short_period))
    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['MA_long'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        if oi == 0:
            if kd2['MA_short'][n-1] <= kd2['MA_long'][n-1] and kd2['MA_short'][n] > kd2['MA_long'][n]:
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif kd2['MA_short'][n-1] >= kd2['MA_long'][n-1] and kd2['MA_short'][n] < kd2['MA_long'][n]:
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl:
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl:
                _cover_all_short(rec, kd2, n)
    return rec


def run_RSI_trend(kd, long_period=10, short_period=5, stop_loss=30.0, order_qty=1):
    """(二) RSI 順勢策略：短RSI上穿長RSI且RSI>50買入，下穿且RSI<50賣出。"""
    rec = Record()
    kd2 = dict(kd)
    kd2['RSI_long']  = RSI(kd2, timeperiod=int(long_period))
    kd2['RSI_short'] = RSI(kd2, timeperiod=int(short_period))
    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['RSI_long'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        if oi == 0:
            if (kd2['RSI_short'][n-1] <= kd2['RSI_long'][n-1] and
                    kd2['RSI_short'][n] > kd2['RSI_long'][n] and kd2['RSI_long'][n] > 50):
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif (kd2['RSI_short'][n-1] >= kd2['RSI_long'][n-1] and
                    kd2['RSI_short'][n] < kd2['RSI_long'][n] and kd2['RSI_long'][n] < 50):
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl:
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl:
                _cover_all_short(rec, kd2, n)
    return rec


def run_RSI_reversal(kd, period=5, ceil=80, floor=20, stop_loss=30.0, order_qty=1):
    """(三) RSI 逆勢策略：RSI從超賣區反彈買入，從超買區回落賣出。"""
    rec = Record()
    kd2 = dict(kd)
    kd2['RSI'] = RSI(kd2, timeperiod=int(period))
    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['RSI'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        if oi == 0:
            if kd2['RSI'][n-1] <= floor and kd2['RSI'][n] > floor:
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif kd2['RSI'][n-1] >= ceil and kd2['RSI'][n] < ceil:
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl or kd2['RSI'][n] > ceil:
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl or kd2['RSI'][n] < floor:
                _cover_all_short(rec, kd2, n)
    return rec


def run_BBands(kd, period=60, nbdevup=2.0, nbdevdn=2.0, stop_loss=30.0, order_qty=1):
    """(四) 布林通道策略：價格從下軌反彈買入，從上軌回落賣出。"""
    rec = Record()
    kd2 = dict(kd)
    kd2['Upper'], kd2['Middle'], kd2['Lower'] = BBANDS(
        kd2, timeperiod=int(period), nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['Middle'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        if oi == 0:
            if kd2['close'][n-1] <= kd2['Lower'][n-1] and kd2['close'][n] > kd2['Lower'][n]:
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif kd2['close'][n-1] >= kd2['Upper'][n-1] and kd2['close'][n] < kd2['Upper'][n]:
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl or kd2['close'][n] >= kd2['Upper'][n]:
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl or kd2['close'][n] <= kd2['Lower'][n]:
                _cover_all_short(rec, kd2, n)
    return rec


def run_MACD(kd, fast=12, slow=26, signal=9, stop_loss=30.0, order_qty=1):
    """
    (五) MACD 策略：
    - 買進：DIF 上穿 DEA（macdhist 由負轉正），且 DIF > 0（多頭確認）
    - 賣出：DIF 下穿 DEA（macdhist 由正轉負），且 DIF < 0（空頭確認）
    - 在倉期間若出現反向交叉則平倉，配合移動停損
    """
    rec = Record()
    kd2 = dict(kd)
    kd2['macd'], kd2['macdsignal'], kd2['macdhist'] = MACD(
        kd2, fastperiod=int(fast), slowperiod=int(slow), signalperiod=int(signal))
    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['macdhist'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        # 黃金交叉：macdhist 由負轉正，且 DIF(macd) > 0
        golden = kd2['macdhist'][n-1] < 0 and kd2['macdhist'][n] >= 0 and kd2['macd'][n] > 0
        # 死亡交叉：macdhist 由正轉負，且 DIF(macd) < 0
        death  = kd2['macdhist'][n-1] > 0 and kd2['macdhist'][n] <= 0 and kd2['macd'][n] < 0

        if oi == 0:
            if golden:
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif death:
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl or death:      # 移動停損 或 死亡交叉出場
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl or golden:     # 移動停損 或 黃金交叉出場
                _cover_all_short(rec, kd2, n)
    return rec


def run_KDJ(kd, fastk=5, slowk=3, slowd=3, oversold=20, overbought=80, stop_loss=30.0, order_qty=1):
    """
    (六) KDJ 策略：
    - 買進：K 線由下往上穿越 D 線（黃金交叉），且 K 值在超賣區以下（< overbought）
    - 賣出：K 線由上往下穿越 D 線（死亡交叉），且 K 值在超買區以上（> oversold）
    - 在倉期間若出現反向交叉則平倉，配合移動停損
    """
    rec = Record()
    kd2 = dict(kd)
    stoch = STOCH(kd2, fastk_period=int(fastk), slowk_period=int(slowk),
                  slowk_matype=0, slowd_period=int(slowd), slowd_matype=0)
    if isinstance(stoch, (list, tuple)):
        kd2['slowk'] = np.asarray(stoch[0], dtype=np.float64)
        kd2['slowd'] = np.asarray(stoch[1], dtype=np.float64)
    else:
        kd2['slowk'] = np.asarray(stoch['slowk'], dtype=np.float64)
        kd2['slowd'] = np.asarray(stoch['slowd'], dtype=np.float64)
    kd2['J'] = 3 * kd2['slowk'] - 2 * kd2['slowd']

    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['slowk'][n-1]) or np.isnan(kd2['slowd'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        # K 上穿 D（黃金交叉）
        k_cross_up   = kd2['slowk'][n-1] <= kd2['slowd'][n-1] and kd2['slowk'][n] > kd2['slowd'][n]
        # K 下穿 D（死亡交叉）
        k_cross_down = kd2['slowk'][n-1] >= kd2['slowd'][n-1] and kd2['slowk'][n] < kd2['slowd'][n]

        if oi == 0:
            if k_cross_up and kd2['slowk'][n] < overbought:
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif k_cross_down and kd2['slowk'][n] > oversold:
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl or k_cross_down:  # 移動停損 或 死亡交叉出場
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl or k_cross_up:    # 移動停損 或 黃金交叉出場
                _cover_all_short(rec, kd2, n)
    return rec


def run_custom(kd, ma_period=20, rsi_period=14, rsi_buy=30, rsi_sell=70, stop_loss=30.0, order_qty=1):
    """
    (七) 自定義策略 - RSI 超賣/超買 + MA 趨勢過濾：
    - 買進：RSI 從超賣區回升（RSI 上穿 rsi_buy），且收盤 > MA（上升趨勢確認）
    - 賣出：RSI 從超買區回落（RSI 下穿 rsi_sell），且收盤 < MA（下降趨勢確認）
    """
    rec = Record()
    kd2 = dict(kd)
    kd2['MA']  = SMA(kd2, timeperiod=int(ma_period))
    kd2['RSI'] = RSI(kd2, timeperiod=int(rsi_period))
    sl = 0.0
    for n in range(1, len(kd2['time']) - 1):
        if np.isnan(kd2['MA'][n]) or np.isnan(kd2['RSI'][n-1]):
            continue
        oi = rec.GetOpenInterest()
        if oi == 0:
            if kd2['RSI'][n-1] <= rsi_buy and kd2['RSI'][n] > rsi_buy and kd2['close'][n] > kd2['MA'][n]:
                rec.Order('Buy', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] - stop_loss
            elif kd2['RSI'][n-1] >= rsi_sell and kd2['RSI'][n] < rsi_sell and kd2['close'][n] < kd2['MA'][n]:
                rec.Order('Sell', kd2['product'][n+1], kd2['time'][n+1], kd2['open'][n+1], order_qty)
                sl = kd2['open'][n+1] + stop_loss
        elif oi > 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Sell', kd2['product'][n], kd2['time'][n], kd2['close'][n], rec.GetOpenInterest()); continue
            sl = max(sl, kd2['close'][n] - stop_loss)
            if kd2['close'][n] < sl or kd2['RSI'][n] > rsi_sell:
                _cover_all_long(rec, kd2, n)
        elif oi < 0:
            if kd2['product'][n+1] != kd2['product'][n]:
                rec.Cover('Buy', kd2['product'][n], kd2['time'][n], kd2['close'][n], -rec.GetOpenInterest()); continue
            sl = min(sl, kd2['close'][n] + stop_loss)
            if kd2['close'][n] > sl or kd2['RSI'][n] < rsi_buy:
                _cover_all_short(rec, kd2, n)
    return rec


# 策略名稱 → 函數對應表
STRATEGIES = {
    "(一) 移動平均線 (MA)":      run_MA,
    "(二) RSI 順勢":             run_RSI_trend,
    "(三) RSI 逆勢":             run_RSI_reversal,
    "(四) 布林通道 (BBands)":    run_BBands,
    "(五) MACD":                run_MACD,
    "(六) KDJ":                 run_KDJ,
    "(七) 自定義 (RSI+MA)":      run_custom,
}


# ─────────────────────────────────────────────────────────────────────────────
# 顯示績效指標
# ─────────────────────────────────────────────────────────────────────────────
def show_metrics(rec: Record):
    profits = rec.GetProfit()
    n = rec.GetTotalNumber()
    sharpe = float(np.mean(profits) / (np.std(profits) + 1e-10)) if n > 1 else 0.0
    calmar = float(rec.GetTotalProfit() / (rec.GetMDD() + 1e-10)) if rec.GetMDD() > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("淨利 (點)", f"{rec.GetTotalProfit():.2f}")
    c1.metric("交易次數", n)
    c2.metric("勝率", f"{rec.GetWinRate():.1%}")
    c2.metric("平均損益", f"{rec.GetAverageProfit():.2f}")
    c3.metric("最大連續虧損", f"{rec.GetAccLoss():.2f}")
    c3.metric("MDD (點)", f"{rec.GetMDD():.2f}")
    c4.metric("Sharpe Ratio", f"{sharpe:.3f}")
    c4.metric("Calmar Ratio", f"{calmar:.3f}")


def plot_cumulative_profit(rec: Record, title='累計損益'):
    profits = rec.GetCumulativeProfit()
    color = 'green' if profits[-1] >= 0 else 'crimson'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=profits, mode='lines', name='累計損益',
        line=dict(color=color, width=2),
        fill='tozeroy', fillcolor=f'rgba(0,128,0,0.1)' if profits[-1] >= 0 else 'rgba(220,20,60,0.1)'
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(
        title=title, xaxis_title='交易序號', yaxis_title='累計損益 (點)',
        height=350, margin=dict(l=40, r=20, t=50, b=40),
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_kbar(kd: dict):
    df = pd.DataFrame({
        'Open':   kd['open'],
        'High':   kd['high'],
        'Low':    kd['low'],
        'Close':  kd['close'],
        'Volume': kd['volume'],
    }, index=pd.DatetimeIndex(kd['time']))
    buf = io.BytesIO()
    mpf.plot(df, type='candle', style='charles', volume=True,
             title='K棒圖', savefig=dict(fname=buf, dpi=100, bbox_inches='tight'))
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 參數最佳化（Grid Search）
# ─────────────────────────────────────────────────────────────────────────────
def calc_score(rec: Record, alpha: float = 0.5) -> float:
    """目標函數：alpha × Sharpe + (1-alpha) × Calmar"""
    profits = rec.GetProfit()
    if len(profits) < 2:
        return -999.0
    sharpe = float(np.mean(profits) / (np.std(profits) + 1e-10))
    calmar = float(rec.GetTotalProfit() / (rec.GetMDD() + 1e-10)) if rec.GetMDD() > 0 else 0.0
    return alpha * sharpe + (1 - alpha) * calmar


def grid_search(run_func, kd: dict, param_grid: dict, alpha: float = 0.5):
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    best_score, best_params, rows = -np.inf, None, []
    total = 1
    for v in values:
        total *= len(v)

    bar = st.progress(0, text="搜索中...")
    done = 0
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            rec   = run_func(kd, **params)
            score = calc_score(rec, alpha)
            rows.append({**params,
                         '分數':    round(score, 4),
                         '淨利':    round(rec.GetTotalProfit(), 2),
                         '勝率':    f"{rec.GetWinRate():.1%}",
                         'MDD':     round(rec.GetMDD(), 2),
                         '交易次數': rec.GetTotalNumber()})
            if score > best_score:
                best_score, best_params = score, params
        except Exception:
            pass
        done += 1
        bar.progress(done / total, text=f"搜索中... {done}/{total}")
    bar.empty()
    return best_params, best_score, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# AI 評估（Claude API）
# ─────────────────────────────────────────────────────────────────────────────
def ai_evaluate(api_key: str, summary_text: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""你是一位專業的量化交易分析師。以下是多種程式交易策略的回測績效數據，請進行詳細分析與比較：

{summary_text}

請用繁體中文，從以下六個角度提供分析報告：

## 1. 各策略綜合排名
（同時考慮報酬、勝率、Sharpe Ratio、Calmar Ratio 等指標）

## 2. 各策略優缺點
（分別說明每個策略的強項和弱點）

## 3. 風險控管分析
（解讀 MDD、Sharpe Ratio、Calmar Ratio 的意義，哪個策略風險最低？）

## 4. 市場環境適用性
（哪個策略適合趨勢市場？哪個適合震盪市場？）

## 5. 改進建議
（如何調整參數或組合多個策略？）

## 6. 最終推薦
（根據本資料集的特性，推薦哪個策略或策略組合，並說明理由）"""

    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 主介面
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="程式交易回測系統", page_icon="📈", layout="wide")
st.title("📈 程式交易回測系統")
st.caption("策略：移動平均 ｜ RSI順勢 ｜ RSI逆勢 ｜ 布林通道 ｜ MACD ｜ KDJ ｜ 自定義")

# ── 側邊欄：資料載入 ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 資料設定")
    data_source = st.radio("資料來源", ["上傳 Excel", "上傳 SQLite DB", "使用預設 Excel"])

    df_raw = None

    if data_source == "上傳 Excel":
        f = st.file_uploader("選擇 Excel 檔", type=["xlsx", "xls"])
        if f:
            df_raw = pd.read_excel(f, index_col=0)
            st.success(f"已讀取 {len(df_raw)} 筆資料")

    elif data_source == "上傳 SQLite DB":
        f = st.file_uploader("選擇 SQLite 資料庫", type=["db", "sqlite", "sqlite3"])
        if f:
            tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_upload.db")
            with open(tmp, "wb") as out:
                out.write(f.read())
            conn = sqlite3.connect(tmp)
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
            if tables:
                chosen = st.selectbox("選擇資料表", tables)
                df_raw = pd.read_sql(f'SELECT * FROM "{chosen}"', conn)
                st.success(f"已讀取 {len(df_raw)} 筆資料")
            else:
                st.error("資料庫中找不到資料表")
            conn.close()

    else:  # 預設 Excel
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "kbars_2330_2022-01-01-2024-04-09.xlsx")
        if os.path.exists(default):
            df_raw = pd.read_excel(default, index_col=0)
            st.success(f"已載入預設 Excel（{len(df_raw)} 筆）")
        else:
            st.warning("找不到預設 Excel，請上傳資料檔案")

    kbar_cycle = st.number_input("K棒週期 (分鐘)",
                                  min_value=1, max_value=10080, value=2880, step=1,
                                  help="2880 = 2天 ｜ 1440 = 1天 ｜ 60 = 1小時")
    st.divider()
    st.header("🤖 AI 評估")
    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-...",
                             help="輸入後才能使用「AI 評估」分頁")

# ── 資料前處理 ───────────────────────────────────────────────────────────────
kd = None
if df_raw is not None:
    with st.spinner("處理資料中..."):
        try:
            kd = build_kbar_dic(df_raw, int(kbar_cycle))
            st.sidebar.info(f"K棒數量：{len(kd['time'])} 根\n"
                            f"起：{pd.Timestamp(kd['time'][0]).date()}\n"
                            f"迄：{pd.Timestamp(kd['time'][-1]).date()}")
        except Exception as e:
            st.sidebar.error(f"資料處理失敗：{e}")

if kd is None:
    st.info("👈 請先在左側載入資料（Excel 或 SQLite）")
    st.stop()

# ── 分頁 ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 K棒圖", "📈 策略回測", "🔧 參數最佳化", "🤖 AI 評估"])


# ══════════════════════════════════════════════════════════════════
# Tab 1 — K棒圖
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("K棒圖")
    plot_kbar(kd)
    st.write(f"資料範圍：{pd.Timestamp(kd['time'][0]).date()} ～ "
             f"{pd.Timestamp(kd['time'][-1]).date()}，共 {len(kd['time'])} 根")


# ══════════════════════════════════════════════════════════════════
# Tab 2 — 策略回測
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("策略回測")
    chosen = st.selectbox("選擇策略", list(STRATEGIES.keys()), key="bt_strat")

    with st.expander("📐 策略參數設定", expanded=True):
        if chosen == "(一) 移動平均線 (MA)":
            p1 = st.slider("長期 MA 期數", 5, 100, 10, key="ma_long")
            p2 = st.slider("短期 MA 期數", 1, 30, 2, key="ma_short")
            p3 = st.slider("移動停損 (點)", 1.0, 200.0, 10.0, 1.0, key="ma_sl")
            p4 = st.slider("下單數量", 1, 10, 3, key="ma_qty")
            params = dict(long_period=p1, short_period=p2, stop_loss=p3, order_qty=p4)

        elif chosen == "(二) RSI 順勢":
            p1 = st.slider("長期 RSI 期數", 5, 50, 10, key="rsit_long")
            p2 = st.slider("短期 RSI 期數", 2, 20, 5, key="rsit_short")
            p3 = st.slider("移動停損 (點)", 1.0, 200.0, 30.0, 1.0, key="rsit_sl")
            p4 = st.slider("下單數量", 1, 10, 3, key="rsit_qty")
            params = dict(long_period=p1, short_period=p2, stop_loss=p3, order_qty=p4)

        elif chosen == "(三) RSI 逆勢":
            p1 = st.slider("RSI 期數", 2, 30, 5, key="rsir_p")
            p2 = st.slider("超買線 (Ceil)", 60, 95, 80, key="rsir_ceil")
            p3 = st.slider("超賣線 (Floor)", 5, 40, 20, key="rsir_floor")
            p4 = st.slider("移動停損 (點)", 1.0, 200.0, 30.0, 1.0, key="rsir_sl")
            p5 = st.slider("下單數量", 1, 10, 3, key="rsir_qty")
            params = dict(period=p1, ceil=p2, floor=p3, stop_loss=p4, order_qty=p5)

        elif chosen == "(四) 布林通道 (BBands)":
            p1 = st.slider("布林通道期數", 10, 120, 60, key="bb_p")
            p2 = st.slider("上軌倍數 (σ)", 0.5, 4.0, 2.0, 0.5, key="bb_up")
            p3 = st.slider("下軌倍數 (σ)", 0.5, 4.0, 2.0, 0.5, key="bb_dn")
            p4 = st.slider("移動停損 (點)", 1.0, 200.0, 30.0, 1.0, key="bb_sl")
            p5 = st.slider("下單數量", 1, 10, 1, key="bb_qty")
            params = dict(period=p1, nbdevup=p2, nbdevdn=p3, stop_loss=p4, order_qty=p5)

        elif chosen == "(五) MACD":
            p1 = st.slider("快線期數 (Fast EMA)", 5, 30, 12, key="macd_fast")
            p2 = st.slider("慢線期數 (Slow EMA)", 10, 60, 26, key="macd_slow")
            p3 = st.slider("訊號線期數 (Signal)", 3, 20, 9, key="macd_sig")
            p4 = st.slider("移動停損 (點)", 1.0, 200.0, 30.0, 1.0, key="macd_sl")
            p5 = st.slider("下單數量", 1, 10, 1, key="macd_qty")
            params = dict(fast=p1, slow=p2, signal=p3, stop_loss=p4, order_qty=p5)
            st.info("買進：DIF上穿DEA（黃金交叉）且DIF>0 ｜ 賣出：DIF下穿DEA（死亡交叉）且DIF<0")

        elif chosen == "(六) KDJ":
            p1 = st.slider("RSV 期數 (fastk)", 3, 20, 5, key="kdj_fk")
            p2 = st.slider("K 值平滑期數 (slowk)", 1, 10, 3, key="kdj_sk")
            p3 = st.slider("D 值平滑期數 (slowd)", 1, 10, 3, key="kdj_sd")
            p4 = st.slider("超賣線", 10, 40, 20, key="kdj_os")
            p5 = st.slider("超買線", 60, 90, 80, key="kdj_ob")
            p6 = st.slider("移動停損 (點)", 1.0, 200.0, 30.0, 1.0, key="kdj_sl")
            p7 = st.slider("下單數量", 1, 10, 1, key="kdj_qty")
            params = dict(fastk=p1, slowk=p2, slowd=p3,
                          oversold=p4, overbought=p5, stop_loss=p6, order_qty=p7)
            st.info("買進：K上穿D（黃金交叉）且K<超買線 ｜ 賣出：K下穿D（死亡交叉）且K>超賣線")

        else:  # 自定義
            p1 = st.slider("MA 趨勢期數", 5, 100, 20, key="cus_ma")
            p2 = st.slider("RSI 期數", 5, 30, 14, key="cus_rsi")
            p3 = st.slider("RSI 買進門檻 (超賣)", 10, 50, 30, key="cus_rb")
            p4 = st.slider("RSI 賣出門檻 (超買)", 50, 90, 70, key="cus_rs")
            p5 = st.slider("移動停損 (點)", 1.0, 200.0, 30.0, 1.0, key="cus_sl")
            p6 = st.slider("下單數量", 1, 10, 1, key="cus_qty")
            params = dict(ma_period=p1, rsi_period=p2,
                          rsi_buy=p3, rsi_sell=p4, stop_loss=p5, order_qty=p6)
            st.info("買進：RSI從超賣區回升 且 收盤>MA ｜ 賣出：RSI從超買區回落 且 收盤<MA")

    if st.button("▶ 執行回測", type="primary", key="run_bt"):
        with st.spinner("回測中..."):
            try:
                rec = STRATEGIES[chosen](kd, **params)
                if rec.GetTotalNumber() == 0:
                    st.warning("此參數組合沒有產生任何交易，請調整參數")
                else:
                    st.success(f"回測完成！共 {rec.GetTotalNumber()} 筆交易")
                    show_metrics(rec)
                    plot_cumulative_profit(rec, f"{chosen} 累計損益曲線")

                    with st.expander("📋 查看完整交易紀錄"):
                        tr = rec.GetTradeRecord()
                        if tr:
                            tr_df = pd.DataFrame(tr, columns=[
                                '方向', '商品', '進場時間', '進場價', '出場時間', '出場價', '數量'])
                            tr_df['損益'] = tr_df.apply(
                                lambda r: (r['出場價'] - r['進場價']) * r['數量']
                                if r['方向'] == 'B'
                                else (r['進場價'] - r['出場價']) * r['數量'], axis=1)
                            tr_df['方向'] = tr_df['方向'].map({'B': '買多', 'S': '賣空'})
                            st.dataframe(tr_df, use_container_width=True)
            except Exception as e:
                st.error(f"回測失敗：{e}")


# ══════════════════════════════════════════════════════════════════
# Tab 3 — 參數最佳化
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔧 參數最佳化（Grid Search）")
    st.info("目標函數：**α × Sharpe Ratio + (1−α) × Calmar Ratio**（同時考量報酬與風險）")

    opt_strat = st.selectbox("選擇要最佳化的策略", list(STRATEGIES.keys()), key="opt_s")
    alpha_w   = st.slider("α 權重（Sharpe vs Calmar）", 0.0, 1.0, 0.5, 0.1,
                           help="1.0 = 純Sharpe，0.0 = 純Calmar，0.5 = 各半")

    with st.expander("設定參數搜索範圍", expanded=True):
        if opt_strat == "(一) 移動平均線 (MA)":
            lr = st.multiselect("長期 MA 期數", list(range(5, 51, 5)), [10, 20, 30])
            sr = st.multiselect("短期 MA 期數", list(range(1, 16, 2)), [1, 5])
            slr = st.multiselect("停損 (點)", [5, 10, 20, 30, 50], [10, 20])
            param_grid = dict(long_period=lr, short_period=sr, stop_loss=slr, order_qty=[3])

        elif opt_strat == "(二) RSI 順勢":
            lr = st.multiselect("長期 RSI", list(range(5, 31, 5)), [10, 15, 20])
            sr = st.multiselect("短期 RSI", list(range(2, 11, 2)), [2, 4])
            slr = st.multiselect("停損 (點)", [10, 20, 30, 50], [20, 30])
            param_grid = dict(long_period=lr, short_period=sr, stop_loss=slr, order_qty=[3])

        elif opt_strat == "(三) RSI 逆勢":
            pr  = st.multiselect("RSI 期數", list(range(3, 16, 2)), [5, 7, 9])
            cr  = st.multiselect("超買線", [70, 75, 80, 85], [75, 80])
            fr  = st.multiselect("超賣線", [15, 20, 25, 30], [20, 25])
            slr = st.multiselect("停損 (點)", [10, 20, 30, 50], [20, 30])
            param_grid = dict(period=pr, ceil=cr, floor=fr, stop_loss=slr, order_qty=[3])

        elif opt_strat == "(四) 布林通道 (BBands)":
            pr   = st.multiselect("布林期數", [20, 40, 60, 80], [40, 60])
            upr  = st.multiselect("上軌倍數", [1.5, 2.0, 2.5, 3.0], [2.0, 2.5])
            slr  = st.multiselect("停損 (點)", [10, 20, 30, 50], [20, 30])
            param_grid = dict(period=pr, nbdevup=upr, nbdevdn=[2.0], stop_loss=slr, order_qty=[1])

        elif opt_strat == "(五) MACD":
            fsr  = st.multiselect("快線期數", [8, 10, 12, 15], [10, 12])
            slwr = st.multiselect("慢線期數", [20, 26, 30], [20, 26])
            sigr = st.multiselect("訊號線期數", [7, 9, 12], [7, 9])
            slr  = st.multiselect("停損 (點)", [10, 20, 30, 50], [20, 30])
            param_grid = dict(fast=fsr, slow=slwr, signal=sigr, stop_loss=slr, order_qty=[1])

        elif opt_strat == "(六) KDJ":
            fkr  = st.multiselect("RSV 期數 (fastk)", [3, 5, 9], [5, 9])
            skr  = st.multiselect("K 平滑期數", [1, 3, 5], [3])
            sdr  = st.multiselect("D 平滑期數", [1, 3, 5], [3])
            osr  = st.multiselect("超賣線", [15, 20, 25], [20])
            obr  = st.multiselect("超買線", [75, 80, 85], [80])
            slr  = st.multiselect("停損 (點)", [10, 20, 30, 50], [20, 30])
            param_grid = dict(fastk=fkr, slowk=skr, slowd=sdr,
                              oversold=osr, overbought=obr, stop_loss=slr, order_qty=[1])

        else:  # 自定義
            mar  = st.multiselect("MA 期數", [10, 20, 30, 50], [20])
            rsir = st.multiselect("RSI 期數", [7, 14, 21], [14])
            rbr  = st.multiselect("RSI 買進", [25, 30, 35], [30])
            rsr  = st.multiselect("RSI 賣出", [65, 70, 75], [70])
            slr  = st.multiselect("停損 (點)", [10, 20, 30, 50], [20, 30])
            param_grid = dict(ma_period=mar, rsi_period=rsir,
                              rsi_buy=rbr, rsi_sell=rsr, stop_loss=slr, order_qty=[1])

    total_combos = 1
    for v in param_grid.values():
        total_combos *= max(len(v), 1)
    st.write(f"預計搜索組合數：**{total_combos}**")

    if st.button("🚀 開始最佳化", type="primary", key="run_opt"):
        if any(len(v) == 0 for v in param_grid.values()):
            st.error("請確保每個參數至少選擇一個值")
        else:
            best_p, best_s, df_res = grid_search(
                STRATEGIES[opt_strat], kd, param_grid, alpha_w)

            if best_p:
                st.success(f"最佳化完成！最高分數：{best_s:.4f}")
                st.subheader("最佳參數組合")
                st.json(best_p)

                st.subheader("最佳參數回測結果")
                rec_best = STRATEGIES[opt_strat](kd, **best_p)
                show_metrics(rec_best)
                plot_cumulative_profit(rec_best, f"{opt_strat} 最佳化後累計損益")

                st.subheader("所有組合結果（依分數排序）")
                st.dataframe(
                    df_res.sort_values('分數', ascending=False).reset_index(drop=True),
                    use_container_width=True)
            else:
                st.warning("沒有找到有效組合，請擴大參數範圍")


# ══════════════════════════════════════════════════════════════════
# Tab 4 — AI 評估
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 生成式 AI 策略評估報告")
    st.write("使用 **Claude** 自動分析並比較各策略績效，生成專業量化分析報告。")

    if not api_key:
        st.warning("⚠️ 請先在左側側邊欄輸入 Anthropic API Key")
        st.stop()

    selected_strats = st.multiselect(
        "選擇要評估的策略（可多選）",
        list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()))

    if st.button("🤖 生成 AI 評估報告", type="primary", key="run_ai"):
        if not selected_strats:
            st.error("請至少選擇一個策略")
        else:
            with st.spinner("回測所有策略並生成 AI 分析報告（約需 30 秒）..."):
                # 收集各策略預設參數的回測結果
                summary_lines = []
                cols = st.columns(len(selected_strats))
                for i, sname in enumerate(selected_strats):
                    try:
                        rec = STRATEGIES[sname](kd)
                        profits = rec.GetProfit()
                        n = rec.GetTotalNumber()
                        sharpe = float(np.mean(profits) / (np.std(profits) + 1e-10)) if n > 1 else 0
                        calmar = float(rec.GetTotalProfit() / (rec.GetMDD() + 1e-10)) if rec.GetMDD() > 0 else 0
                        line = (f"【{sname}】"
                                f"淨利={rec.GetTotalProfit():.2f}點, "
                                f"交易次數={n}, "
                                f"勝率={rec.GetWinRate():.1%}, "
                                f"平均損益={rec.GetAverageProfit():.2f}, "
                                f"MDD={rec.GetMDD():.2f}, "
                                f"Sharpe={sharpe:.3f}, "
                                f"Calmar={calmar:.3f}")
                        summary_lines.append(line)
                        cols[i].metric(sname.split(')')[0]+')', f"{rec.GetTotalProfit():.1f}")
                    except Exception as e:
                        summary_lines.append(f"【{sname}】計算失敗：{e}")

                summary_text = "\n".join(summary_lines)

                st.subheader("各策略績效摘要")
                st.code(summary_text, language=None)

                try:
                    report = ai_evaluate(api_key, summary_text)
                    st.subheader("AI 分析報告")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI 評估失敗：{e}\n\n請確認 API Key 是否正確。")
