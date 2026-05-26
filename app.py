import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time

# ---------------------------------------------------------
# 1. 全局配置與資料庫 (選股池與名稱字典)
# ---------------------------------------------------------
st.set_page_config(page_title="AI 終極量化選股系統", layout="wide")
st.title("🧠 終極版 v4.1：多群組 AI 選股與動態回測整合平台")
st.markdown("資深投資人專屬：一鍵掃描不同 ETF 族群，並自動帶出股票名稱，精準鎖定高潛力飆股。")

# 🏆 挑戰 1 & 2 解法：建立不同的選股池字典 (代號: 名稱)
POOL_0050 = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達", "2308.TW": "台達電",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2412.TW": "中華電", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "3231.TW": "緯創", "1216.TW": "統一", "2002.TW": "中鋼", "3711.TW": "日月光", "2884.TW": "玉山金"
}

POOL_HIGH_DIV = {
    "3231.TW": "緯創", "2357.TW": "華碩", "2356.TW": "英業達", "2382.TW": "廣達", "2324.TW": "仁寶",
    "2379.TW": "瑞昱", "3034.TW": "聯詠", "2449.TW": "京元電子", "2912.TW": "統一超", "2890.TW": "永豐金"
}

POOL_CUSTOM = { 
    "2330.TW": "台積電", "2454.TW": "聯發科", "2354.TW": "鴻準", "2382.TW": "廣達", "6462.TWO": "神盾",
    "1519.TW": "華城", "7734.TWO": "印能", "2317.TW": "鴻海", "3231.TW": "緯創", "2449.TW": "京元電子",
    "3019.TW": "亞光"
}

AI_FEATURES = ['Volatility', 'Dist_SMA', 'Vol_Ratio', 'RSI', 'OBV_Trend']

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

# ---------------------------------------------------------
# 2. 公用函式庫
# ---------------------------------------------------------
@st.cache_data 
def load_data(sym):
    stock = yf.Ticker(sym, session=session)
    df = stock.history(period="3y")[['High', 'Low', 'Close', 'Volume']].dropna()
    return df

def add_features(df):
    df_feat = df.copy()
    df_feat['Daily_Ret'] = df_feat['Close'].pct_change()
    df_feat['Volatility'] = df_feat['Daily_Ret'].rolling(10).std()
    df_feat['SMA_20'] = df_feat['Close'].rolling(20).mean()
    df_feat['Dist_SMA'] = (df_feat['Close'] - df_feat['SMA_20']) / df_feat['SMA_20']
    df_feat['Vol_Ratio'] = df_feat['Volume'] / df_feat['Volume'].rolling(5).mean()
    
    delta = df_feat['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df_feat['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df_feat['Direction'] = np.sign(df_feat['Close'].diff())
    df_feat['OBV'] = (df_feat['Direction'] * df_feat['Volume']).fillna(0).cumsum()
    df_feat['OBV_Trend'] = np.where(df_feat['OBV'] > df_feat['OBV'].rolling(20).mean(), 1, -1)
    return df_feat

# ---------------------------------------------------------
# 3. 回測引擎模組
# ---------------------------------------------------------
@st.cache_data
def train_ai_model(df):
    df_ai = add_features(df)
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
# 4. AI 市場選股掃描器 (預測漲幅)
# ---------------------------------------------------------
def scan_single_stock(symbol, stock_name):
    try:
        stock = yf.Ticker(symbol, session=session)
        df = stock.history(period="2y")[['High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) < 100: return None
        
        df = add_features(df)
        df['Target_PCT'] = (df['Close'].shift(-5) - df['Close']) / df['Close'] * 100
        
        today_data = df.iloc[-1:]
        train_data = df.iloc[:-1].dropna()
        
        X = train_data[AI_FEATURES]
        y = train_data['Target_PCT']
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        X_today = today_data[AI_FEATURES]
        predicted_pct_change = model.predict(X_today)[0]
        
        return {
            'Symbol': f"{symbol} {stock_name}", # 組合代號與名稱
            'Close': round(today_data['Close'].iloc[0], 2),
            'Pred_5D_Pct': round(predicted_pct_change, 2)
        }
    except Exception as e:
        return None

# ---------------------------------------------------------
# 5. 主程式啟動與網頁渲染
# ---------------------------------------------------------
app_mode = st.sidebar.selectbox("🗺️ 模式選擇", ["個股回測 (突破策略)", "市場掃描 (AI 選股)"])

if app_mode == "個股回測 (突破策略)":
    st.sidebar.header("⚙️ 標的與風控設定")
    symbol = st.sidebar.text_input("股票代號 (Yahoo Finance)", value="2317.TW")
    sl_input = st.sidebar.number_input("停損百分比 (%)", min_value=1.0, value=5.0)
    tp_input = st.sidebar.number_input("停利百分比 (%)", min_value=5.0, value=10.0)
    sl_percent, tp_percent = sl_input / 100, tp_input / 100

    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI 機器學習引擎")
    use_ai = st.sidebar.toggle("啟用 AI 真假突破濾網", value=True)

    if st.sidebar.button("🚀 啟動 AI 雙刀流回測", type="primary"):
        with st.spinner(f"正在計算高階特徵並訓練 AI..."):
            df = load_data(symbol)
            model = train_ai_model(df) if use_ai else None
            if use_ai: st.success("🤖 AI 大腦裝載完畢！")
                
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
                st.warning(f"🛡️ **AI 戰功回報：** AI 強制攔截了 **{len(block_d)} 次** 假突破！")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Close'], name='收盤價', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Upper_Band'], name=f'{best_n}日上軌', line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Lower_Band'], name=f'{best_n}日下軌', line=dict(color='green', dash='dot')))
            
            fig.add_trace(go.Scatter(x=b_d, y=b_p, mode='markers', name='✅ AI 允許進場', marker=dict(color='green', symbol='triangle-up', size=14, line=dict(width=1, color='DarkSlateGrey'))))
            fig.add_trace(go.Scatter(x=s_d, y=s_p, mode='markers', name='出場點', marker=dict(color='red', symbol='triangle-down', size=14, line=dict(width=1, color='DarkSlateGrey'))))
            
            if use_ai and len(block_d) > 0:
                fig.add_trace(go.Scatter(x=block_d, y=block_p, mode='markers', name='🚫 高階 AI 攔截之假突破', marker=dict(color='gray', symbol='x', size=12, line=dict(width=2, color='black'))))
            
            fig.update_layout(height=600, hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

elif app_mode == "市場掃描 (AI 選股)":
    st.header("🏆 台股 AI 選股掃描雷達")
    st.markdown("請選擇你要掃描的股票族群。AI 迴歸模型將依據當前的量價與籌碼特徵，**預測未來 5 天的百分比變動幅度**。")
    st.markdown("---")
    
    # 建立三個紅色按鈕
    col1, col2, col3 = st.columns(3)
    scan_0050 = col1.button("🚀 掃描 0050 (大型權值)", type="primary", use_container_width=True)
    scan_div = col2.button("🚀 掃描 高股息ETF", type="primary", use_container_width=True)
    scan_custom = col3.button("🚀 掃描 專屬自選股", type="primary", use_container_width=True)
    
    target_pool = None
    pool_name = ""
    
    if scan_0050:
        target_pool = POOL_0050
        pool_name = "0050 大型權值股"
    elif scan_div:
        target_pool = POOL_HIGH_DIV
        pool_name = "高股息 ETF 成分股"
    elif scan_custom:
        target_pool = POOL_CUSTOM
        pool_name = "專屬自選股"
        
    # 如果有按下任何一個掃描按鈕，就開始執行
    if target_pool is not None:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 使用 items() 同時抓取代號與名稱
        for idx, (sym, name) in enumerate(target_pool.items()):
            progress = (idx + 1) / len(target_pool)
            progress_bar.progress(progress)
            status_text.markdown(f"[{idx+1}/{len(target_pool)}] 正在分析 **{sym} {name}** 的 AI 潛在漲幅...")
            
            res = scan_single_stock(sym, name)
            if res:
                results.append(res)
            
        progress_bar.empty()
        status_text.empty()
        st.success(f"🤖 【{pool_name}】掃描與漲幅預測完成！")
        
        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values(by='Pred_5D_Pct', ascending=False)
        
        top_10 = df_sorted.head(10).reset_index(drop=True)
        top_10.index = top_10.index + 1 
        
        st.subheader(f"🔥 AI 嚴選：{pool_name} 潛在漲幅前 10 名")
        
        def color_pred(val):
            color = 'green' if val > 1 else ('red' if val < -1 else 'black')
            return f'color: {color}; font-weight: bold'

        styled_df = top_10.style.format({'Close': '{:.2f}', 'Pred_5D_Pct': '{:.2f}%'}).map(color_pred, subset=['Pred_5D_Pct'])
        st.dataframe(styled_df, use_container_width=True)