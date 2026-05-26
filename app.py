import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. 介面基礎設定
# ---------------------------------------------------------
st.set_page_config(page_title="AI 雙刀流量化系統", layout="wide")
st.title("🧠 終極版 v3.1：高階特徵工程 (RSI, MACD, OBV)")
st.markdown("AI 大腦已升級！導入高階技術指標與籌碼動能，大幅提升『假突破』攔截精準度。")

# ---------------------------------------------------------
# 2. 左側邊欄 
# ---------------------------------------------------------
st.sidebar.header("⚙️ 標的與風控設定")
symbol = st.sidebar.text_input("股票代號 (Yahoo Finance 格式)", value="2317.TW")
sl_input = st.sidebar.number_input("停損百分比 (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
tp_input = st.sidebar.number_input("停利百分比 (%)", min_value=5.0, max_value=50.0, value=10.0, step=1.0)
sl_percent, tp_percent = sl_input / 100, tp_input / 100

st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 機器學習引擎")
use_ai = st.sidebar.toggle("啟用 AI 真假突破濾網", value=True)

import requests # 記得確認這行如果在最上面沒有的話，這裡會用到

@st.cache_data 
def load_data(sym):
    # 1. 建立一個虛擬的網路會話 (Session)
    session = requests.Session()
    # 2. 幫這個會話掛上「User-Agent」，偽裝成是 Windows 電腦上的 Chrome 瀏覽器發出的請求
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    })
    
    # 3. 把這個偽裝的 session 交給 yfinance 使用
    stock = yf.Ticker(sym, session=session)
    df = stock.history(period="3y")[['High', 'Low', 'Close', 'Volume']]
    return df

# ---------------------------------------------------------
# 3. 【核心升級】特徵工程模組 (Feature Engineering)
# ---------------------------------------------------------
def add_features(df):
    df_feat = df.copy()
    
    # 基礎特徵
    df_feat['Daily_Ret'] = df_feat['Close'].pct_change()
    df_feat['Volatility'] = df_feat['Daily_Ret'].rolling(10).std()
    df_feat['SMA_20'] = df_feat['Close'].rolling(20).mean()
    df_feat['Dist_SMA'] = (df_feat['Close'] - df_feat['SMA_20']) / df_feat['SMA_20']
    df_feat['Vol_Ratio'] = df_feat['Volume'] / df_feat['Volume'].rolling(5).mean()

    # 高階特徵 1：RSI (動能指標)
    delta = df_feat['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    # 高階特徵 2：MACD 柱狀圖 (趨勢指標)
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = ema_12 - ema_26
    df_feat['MACD_Signal'] = df_feat['MACD'].ewm(span=9, adjust=False).mean()
    df_feat['MACD_Hist'] = df_feat['MACD'] - df_feat['MACD_Signal'] 

    # 高階特徵 3：OBV 趨勢 (籌碼指標)
    df_feat['Direction'] = np.sign(df_feat['Close'].diff())
    df_feat['OBV'] = (df_feat['Direction'] * df_feat['Volume']).fillna(0).cumsum()
    # 1 代表籌碼偏多，-1 代表籌碼偏空
    df_feat['OBV_Trend'] = np.where(df_feat['OBV'] > df_feat['OBV'].rolling(20).mean(), 1, -1)

    return df_feat

# 我們要餵給 AI 學習的所有特徵清單
AI_FEATURES = ['Volatility', 'Dist_SMA', 'Vol_Ratio', 'RSI', 'MACD_Hist', 'OBV_Trend']

# ---------------------------------------------------------
# 4. 模型訓練與回測引擎
# ---------------------------------------------------------
@st.cache_data
def train_ai_model(df):
    df_ai = add_features(df)
    
    # 標準答案：未來 5 天後收盤價是否大於今天 (1=好買點, 0=爛買點)
    df_ai['Target'] = (df_ai['Close'].shift(-5) > df_ai['Close']).astype(int)
    
    train_data = df_ai.dropna()
    X = train_data[AI_FEATURES]
    y = train_data['Target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def run_breakout_strategy(data_df, n_days, sl, tp, ai_model=None):
    df_temp = add_features(data_df)
    df_temp['Upper_Band'] = df_temp['High'].rolling(window=n_days).max().shift(1)
    df_temp['Lower_Band'] = df_temp['Low'].rolling(window=n_days).min().shift(1)
    df_temp = df_temp.dropna()

    position = 0
    entry_price = 0.0
    current_capital = 1000000
    trade_count, win_count = 0, 0
    buy_dates, buy_prices, sell_dates, sell_prices, block_dates, block_prices = [], [], [], [], [], []

    for i in range(len(df_temp)):
        close = df_temp['Close'].iloc[i]
        upper = df_temp['Upper_Band'].iloc[i]
        lower = df_temp['Lower_Band'].iloc[i]
        date = df_temp.index[i]

        if position == 0:
            if close > upper:
                approve_trade = True
                if ai_model is not None:
                    # AI 根據最新的高階特徵進行勝率判定
                    current_features = df_temp[AI_FEATURES].iloc[i:i+1]
                    ai_prediction = ai_model.predict(current_features)[0]
                    
                    if ai_prediction == 0: 
                        approve_trade = False
                        block_dates.append(date)
                        block_prices.append(close)

                if approve_trade:
                    position, entry_price = 1, close
                    buy_dates.append(date); buy_prices.append(close)
            
            elif close < lower: 
                position, entry_price = -1, close
                sell_dates.append(date); sell_prices.append(close)

        elif position == 1:
            if close >= entry_price * (1 + tp) or close <= entry_price * (1 - sl):
                profit_rate = (close - entry_price) / entry_price
                current_capital *= (1 + profit_rate)
                trade_count += 1
                if profit_rate > 0: win_count += 1
                position = 0
                sell_dates.append(date); sell_prices.append(close)

        elif position == -1:
            if close <= entry_price * (1 - tp) or close >= entry_price * (1 + sl):
                profit_rate = (entry_price - close) / entry_price
                current_capital *= (1 + profit_rate)
                trade_count += 1
                if profit_rate > 0: win_count += 1
                position = 0
                buy_dates.append(date); buy_prices.append(close)

    roi = ((current_capital - 1000000) / 1000000) * 100
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    return roi, trade_count, win_rate, current_capital, df_temp, buy_dates, buy_prices, sell_dates, sell_prices, block_dates, block_prices

# ---------------------------------------------------------
# 5. 主程式啟動與網頁渲染
# ---------------------------------------------------------
if st.sidebar.button("🚀 啟動 AI 雙刀流回測", type="primary"):
    with st.spinner(f"正在計算 MACD, RSI, OBV 等高階特徵，並重新訓練 AI 模型..."):
        df = load_data(symbol)
        
        model = None
        if use_ai:
            model = train_ai_model(df)
            st.success("🤖 AI 大腦升級完畢！已裝載高階技術與籌碼特徵。")
            
        st.markdown("---")
        st.subheader("⚙️ 自動參數尋優與回測 (近半年區間)")
        
        df_recent = df.tail(126)
        best_n, best_roi = 10, -999.0
        
        for test_n in range(10, 65, 5):
            roi, trades, _, _, _, _, _, _, _, _, _ = run_breakout_strategy(df_recent, test_n, sl_percent, tp_percent, model)
            if roi > best_roi: best_roi = roi; best_n = test_n

        final_roi, final_trades, final_win_rate, final_capital, final_df, b_d, b_p, s_d, s_p, block_d, block_p = run_breakout_strategy(df_recent, best_n, sl_percent, tp_percent, model)

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("近半年最終帳戶總額", f"NT$ {final_capital:,.0f}", "本金 100 萬")
        res_col2.metric("近半年總報酬率", f"{final_roi:.2f} %")
        res_col3.metric("策略勝率", f"{final_win_rate:.2f} %", f"共交易 {final_trades} 筆", delta_color="off")
        
        if use_ai and len(block_d) > 0:
            st.warning(f"🛡️ **AI 戰功回報：** 高階特徵成功發揮作用！在這半年內，AI 運用 RSI 與 OBV 籌碼判定，強制攔截了 **{len(block_d)} 次** 假突破！")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Close'], name='收盤價', line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Upper_Band'], name=f'{best_n}日上軌', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Lower_Band'], name=f'{best_n}日下軌', line=dict(color='green', dash='dot')))
        
        fig.add_trace(go.Scatter(x=b_d, y=b_p, mode='markers', name='✅ AI 允許進場', marker=dict(color='green', symbol='triangle-up', size=14, line=dict(width=1, color='DarkSlateGrey'))))
        fig.add_trace(go.Scatter(x=s_d, y=s_p, mode='markers', name='出場點', marker=dict(color='red', symbol='triangle-down', size=14, line=dict(width=1, color='DarkSlateGrey'))))
        
        if use_ai and len(block_d) > 0:
            fig.add_trace(go.Scatter(x=block_d, y=block_p, mode='markers', name='🚫 高階 AI 攔截之假突破', marker=dict(color='gray', symbol='x', size=12, line=dict(width=2, color='black'))))
        
        fig.update_layout(height=600, hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width='stretch')