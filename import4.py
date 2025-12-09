# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

st.set_page_config(layout="wide", page_title="BTC Analytics Dashboard")

# -----------------------
# Utilities & indicator functions
# -----------------------
@st.cache_data
def load_from_yfinance(ticker="BTC-USD", period="max", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(columns={"Adj Close": "Adj_Close", "Close": "Close"}, inplace=True)
    return df

def parse_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine='python', error_bad_lines=False)
    return df

def ensure_datetime(df, date_col_candidates=["Date", "date", "timestamp", "Datetime", "Time"]):
    # find a date-like column
    for c in date_col_candidates:
        if c in df.columns:
            df['Date'] = pd.to_datetime(df[c], errors='coerce')
            break
    if 'Date' not in df.columns:
        # try to infer
        possible = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if possible:
            df['Date'] = pd.to_datetime(df[possible[0]], errors='coerce')
    if 'Date' not in df.columns:
        # last resort: if index is datetime
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={'index':'Date'})
            else:
                # no date found
                raise KeyError("No date column detected")
        except Exception:
            raise KeyError("No date column detected. Please upload a CSV with a 'Date' column.")
    df = df.sort_values('Date').dropna(subset=['Date']).reset_index(drop=True)
    return df

def standardize_columns(df):
    # rename common columns
    cols = {c: c.lower() for c in df.columns}
    mapping = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ['close', 'adj close', 'adj_close', 'adjclose']:
            mapping[c] = 'Close'
        if cl in ['open']:
            mapping[c] = 'Open'
        if cl in ['high']:
            mapping[c] = 'High'
        if cl in ['low']:
            mapping[c] = 'Low'
        if cl in ['volume', 'vol']:
            mapping[c] = 'Volume'
        if 'market' in cl and 'cap' in cl:
            mapping[c] = 'Market Cap'
    df = df.rename(columns=mapping)
    return df

def compute_basic_metrics(df):
    # fills expected numeric columns and compute returns
    if 'Close' not in df.columns and 'Adj_Close' in df.columns:
        df['Close'] = df['Adj_Close']

    required = ['Close']
    for r in required:
        if r not in df.columns:
            raise KeyError(f"Required column '{r}' not found in data.")

    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['DailyVolatility'] = df['LogReturn'].rolling(window=21).std()  # 1-month ~21 trading days
    df['AnnualVolatility'] = df['LogReturn'].rolling(window=21).std() * np.sqrt(252)
    # Simple drawdown
    df['Cumulative'] = (1 + df['Return']).cumprod().fillna(1)
    df['RollingMax'] = df['Cumulative'].cummax()
    df['Drawdown'] = df['Cumulative'] / df['RollingMax'] - 1
    return df

def moving_averages(df, windows=[7,21,50,100,200]):
    for w in windows:
        df[f"SMA_{w}"] = df['Close'].rolling(window=w).mean()
        df[f"EMA_{w}"] = df['Close'].ewm(span=w, adjust=False).mean()
    return df

def on_balance_volume(df):
    # OBV
    df['OBV'] = 0
    df.loc[df['Close'] > df['Close'].shift(1), 'OBV'] = df['Volume']
    df.loc[df['Close'] < df['Close'].shift(1), 'OBV'] = -df['Volume']
    df['OBV'] = df['OBV'].cumsum()
    # VPT (Volume Price Trend)
    df['VPT'] = (df['Close'].pct_change().fillna(0) * df['Volume']).cumsum()
    return df

def monthly_yearly_performance(df):
    df2 = df.copy()
    df2['Year'] = df2['Date'].dt.year
    df2['Month'] = df2['Date'].dt.month
    # monthly returns
    monthly = df2.set_index('Date').resample('M').agg({'Close':'last'}).assign(MonthlyReturn=lambda x: x['Close'].pct_change()).reset_index()
    yearly = df2.set_index('Date').resample('Y').agg({'Close':'last'}).assign(YearlyReturn=lambda x: x['Close'].pct_change()).reset_index()
    # performance tables
    monthly['Year'] = monthly['Date'].dt.year
    monthly['Month'] = monthly['Date'].dt.month
    return monthly, yearly

def sma_crossover_strategy(df, fast=21, slow=50, initial_cash=10000):
    # Simple backtest: buy when fast SMA crosses above slow SMA, sell when crosses below
    data = df.copy().reset_index(drop=True)
    data['SMA_fast'] = data['Close'].rolling(window=fast).mean()
    data['SMA_slow'] = data['Close'].rolling(window=slow).mean()
    data['Position'] = 0
    data.loc[data['SMA_fast'] > data['SMA_slow'], 'Position'] = 1
    data['Signal'] = data['Position'].diff().fillna(0)
    cash = initial_cash
    position_units = 0
    trades = []
    values = []
    for idx, row in data.iterrows():
        if row['Signal'] == 1:  # buy
            if cash > 0:
                position_units = cash / row['Close']
                trades.append({'Date': row['Date'], 'Type':'Buy', 'Price':row['Close'], 'Units':position_units})
                cash = 0
        elif row['Signal'] == -1:  # sell
            if position_units > 0:
                cash = position_units * row['Close']
                trades.append({'Date': row['Date'], 'Type':'Sell', 'Price':row['Close'], 'Units':position_units})
                position_units = 0
        # portfolio value
        current_value = cash + position_units * row['Close']
        values.append(current_value)
    data['PortfolioValue'] = values
    return data, pd.DataFrame(trades)

def volume_pressure(df):
    # Volume Pressure: difference between buying and selling volume approximated by price change sign * volume
    df['VP'] = np.sign(df['Close'].diff().fillna(0)) * df['Volume']
    df['VP_Cum'] = df['VP'].cumsum()
    return df

# -----------------------
# App UI
# -----------------------
st.title("📊 Bitcoin (BTC) Analytics Dashboard")
st.sidebar.header("Data source")

data_source = st.sidebar.radio("Choose data source", ("Upload CSV", "Fetch from yfinance (BTC-USD)"))

df = pd.DataFrame()
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV (must include Date and Close columns)", type=['csv'])
    if uploaded is not None:
        try:
            raw = parse_uploaded_csv(uploaded)
            raw = standardize_columns(raw)
            raw = ensure_datetime(raw)
            df = raw.copy()
            st.sidebar.success("CSV loaded successfully.")
        except KeyError as e:
            st.sidebar.error(f"{e}")
        except Exception as e:
            st.sidebar.error(f"Failed to parse CSV: {e}")
elif data_source == "Fetch from yfinance (BTC-USD)":
    period = st.sidebar.selectbox("Period", ["1y","5y","max","2y","6mo","1mo"], index=2)
    interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
    with st.spinner("Fetching data from yfinance..."):
        df = load_from_yfinance(ticker="BTC-USD", period=period, interval=interval)
    if df.empty:
        st.sidebar.error("yfinance returned no data.")
    else:
        if 'Date' not in df.columns:
            df = df.rename(columns={df.columns[0]:'Date'}) if 'Date' not in df.columns else df
        df = ensure_datetime(df)
        df = standardize_columns(df)

if df.empty:
    st.info("No data loaded yet — upload a CSV or fetch from yfinance to start.")
    st.stop()

# compute everything
try:
    df = compute_basic_metrics(df)
except KeyError as e:
    st.error(str(e))
    st.stop()

df = moving_averages(df)
if 'Volume' in df.columns:
    df = on_balance_volume(df)
    df = volume_pressure(df)
else:
    st.warning("Volume column not found; volume-based metrics will be disabled.")

monthly, yearly = monthly_yearly_performance(df)

# Sidebar: dashboard selection
st.sidebar.header("Dashboard")
page = st.sidebar.selectbox("Select view", [
    "1 BTC Price Overview",
    "2 Market Cap Analysis",
    "3 Volume Intelligence",
    "4 Volatility & Risk",
    "5 Monthly & Yearly Performance",
    "6 Profile & Buy-Sell Performance",
    "7 Trend & Moving Averages",
    "8 Volume Pressure Dashboard",
    "9 Market Cap Overview Dashboard"
])

# Common plot helper
def plot_time_series(x, y, title="", yaxis_title="", hovertemplate=None):
    fig = px.line(x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title=yaxis_title)
    return fig

# -----------------------
# Page implementations
# -----------------------
if page == "1 BTC Price Overview":
    st.header("1 — BTC Price Overview")
    col1, col2 = st.columns([3,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.update_layout(title="BTC Close Price", xaxis_title="Date", yaxis_title="Price (USD)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        last = df.iloc[-1]
        st.metric("Latest Price (USD)", f"${last['Close']:.2f}", delta=f"{last['Return']*100:.2f}%")
        if 'Market Cap' in df.columns:
            st.metric("Market Cap", f"${last['Market Cap']:,}")
        st.write("Latest Data:")
        st.dataframe(last[['Date','Open','High','Low','Close','Volume']].dropna(axis=1, how='all').astype(str))

    st.markdown("### Price distribution")
    st.plotly_chart(px.histogram(df, x='Close', nbins=60, title="Price Distribution"), use_container_width=True)

if page == "2 Market Cap Analysis":
    st.header("2 — Market Cap Analysis")
    if 'Market Cap' in df.columns:
        fig = px.line(df, x='Date', y='Market Cap', title="Market Capitalization Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Market Cap vs Price")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', yaxis='y1'))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Market Cap'], name='Market Cap', yaxis='y2'))
        fig2.update_layout(yaxis=dict(title='Price (USD)'), yaxis2=dict(title='Market Cap', overlaying='y', side='right'))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No 'Market Cap' column in your data. If you want market cap analysis, upload CSV with a 'Market Cap' column.")

if page == "3 Volume Intelligence":
    st.header("3 — Volume Intelligence")
    if 'Volume' in df.columns:
        st.plotly_chart(px.line(df, x='Date', y='Volume', title="Trading Volume"), use_container_width=True)
        st.markdown("### On-Balance Volume (OBV)")
        st.plotly_chart(px.line(df, x='Date', y='OBV', title="OBV"), use_container_width=True)
        st.markdown("### Volume vs Price scatter")
        st.plotly_chart(px.scatter(df, x='Volume', y='Close', title="Volume vs Price", trendline="ols"), use_container_width=True)
    else:
        st.warning("No Volume data available.")

if page == "4 Volatility & Risk":
    st.header("4 — Volatility & Risk")
    st.markdown("### Annualized Volatility (rolling 21-day)")
    st.plotly_chart(px.line(df, x='Date', y='AnnualVolatility', title="Annualized Volatility"), use_container_width=True)
    st.markdown("### Drawdown")
    fig = px.line(df, x='Date', y='Drawdown', title="Drawdown")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Key risk numbers")
    c1, c2, c3, c4 = st.columns(4)
    cagr = (df['Cumulative'].iloc[-1]) ** (252/len(df)) - 1 if len(df)>0 else np.nan
    max_dd = df['Drawdown'].min()
    recent_vol = df['AnnualVolatility'].iloc[-1]
    sharpe = (df['Return'].mean() / df['Return'].std()) * np.sqrt(252) if df['Return'].std() != 0 else np.nan
    c1.metric("CAGR (approx)", f"{cagr*100:.2f}%" if not np.isnan(cagr) else "N/A")
    c2.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    c3.metric("Recent Annual Vol", f"{recent_vol*100:.2f}%")
    c4.metric("Sharpe (ann.)", f"{sharpe:.2f}")

if page == "5 Monthly & Yearly Performance":
    st.header("5 — Monthly & Yearly Performance")
    st.markdown("### Monthly returns (last 24 months)")
    display_months = monthly.tail(24).copy()
    display_months['MonthlyReturnPct'] = display_months['MonthlyReturn']*100
    st.dataframe(display_months[['Date','Close','MonthlyReturnPct']].rename(columns={'MonthlyReturnPct':'Monthly Return (%)'}).style.format({"Monthly Return (%)":"{:.2f}"}))
    st.markdown("### Yearly returns")
    yearly['YearlyReturnPct'] = yearly['YearlyReturn']*100
    st.dataframe(yearly[['Date','Close','YearlyReturnPct']].rename(columns={'YearlyReturnPct':'Yearly Return (%)'}).style.format({"Yearly Return (%)":"{:.2f}"}))
    # heatmap calendar (pivot)
    heat = monthly.copy()
    heat['Year'] = heat['Date'].dt.year
    heat['Month'] = heat['Date'].dt.month_name().str[:3]
    pivot = heat.pivot(index='Month', columns='Year', values='MonthlyReturn')
    st.markdown("### Monthly returns heatmap")
    fig = px.imshow(pivot.fillna(0).T, labels=dict(x="Month", y="Year", color="Monthly Return"), x=pivot.index, y=pivot.columns)
    st.plotly_chart(fig, use_container_width=True)

if page == "6 Profile & Buy-Sell Performance":
    st.header("6 — Profile & Buy-Sell Performance")
    st.markdown("### Price & a simple SMA crossover strategy backtest")
    fast = st.sidebar.number_input("Fast SMA (days)", value=21, min_value=1)
    slow = st.sidebar.number_input("Slow SMA (days)", value=50, min_value=1)
    initial_cash = st.sidebar.number_input("Initial cash (USD) for backtest", value=10000)
    backtest_df, trades = sma_crossover_strategy(df, fast=fast, slow=slow, initial_cash=initial_cash)
    st.plotly_chart(px.line(backtest_df, x='Date', y=['PortfolioValue','Close'], title="Portfolio Value vs Price"), use_container_width=True)
    st.markdown("#### Trades executed")
    if not trades.empty:
        trades['Price'] = trades['Price'].map("${:,.2f}".format)
        st.dataframe(trades)
        final_value = backtest_df['PortfolioValue'].iloc[-1]
        pnl = final_value - initial_cash
        st.metric("Final Portfolio Value", f"${final_value:,.2f}", delta=f"${pnl:,.2f}")
    else:
        st.write("No trades were triggered by this SMA crossover configuration.")

    st.markdown("### Profile summary")
    st.write("Basic statistics:")
    stats = {
        "Total Days": len(df),
        "Cumulative Return (%)": f"{(df['Cumulative'].iloc[-1]-1)*100:.2f}",
        "Annualized Volatility (%)": f"{df['AnnualVolatility'].iloc[-1]*100:.2f}",
        "Max Drawdown (%)": f"{df['Drawdown'].min()*100:.2f}"
    }
    st.json(stats)

if page == "7 Trend & Moving Averages":
    st.header("7 — Trend & Moving Averages")
    st.markdown("### Price with SMAs & EMAs")
    ma_list = [c for c in df.columns if c.startswith('SMA_') or c.startswith('EMA_')]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(width=1.5)))
    for m in ma_list:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[m], name=m, opacity=0.8))
    fig.update_layout(title="Price with moving averages", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Trend detection (simple slope)")
    window = st.sidebar.slider("Trend slope window (days)", min_value=7, max_value=200, value=50)
    df['Slope'] = df['Close'].diff(window) / df['Close'].shift(window)
    st.plotly_chart(px.line(df, x='Date', y='Slope', title=f"Slope over {window} days"), use_container_width=True)
    st.write("Positive slope indicates uptrend; negative slope indicates downtrend.")

if page == "8 Volume Pressure Dashboard":
    st.header("8 — Volume Pressure Dashboard")
    if 'VP' in df.columns:
        st.plotly_chart(px.line(df, x='Date', y='VP_Cum', title="Cumulative Volume Pressure (approx)"), use_container_width=True)
        st.plotly_chart(px.scatter(df, x='VP', y='Return', title="Volume Pressure vs Returns", trendline='ols'), use_container_width=True)
    else:
        st.warning("Volume Pressure requires Volume and Close columns.")

if page == "9 Market Cap Overview Dashboard":
    st.header("9 — Market Cap Overview Dashboard")
    if 'Market Cap' in df.columns:
        df['MarketCapChange'] = df['Market Cap'].pct_change()
        fig = px.line(df, x='Date', y='Market Cap', title="Market Cap Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Market Cap growth rates")
        st.plotly_chart(px.bar(df.tail(60), x='Date', y='MarketCapChange', title="Recent Market Cap % change (last 60 rows)"), use_container_width=True)
        # dominance if total market cap column exists? try to detect
        st.markdown("### Notes")
        st.write("If you have a 'Total Market Cap' column (e.g., total crypto market cap), add it to your CSV and this section can compute BTC dominance.")
    else:
        st.warning("Market Cap column not present in dataset.")

# -----------------------
# Footer: data download & raw view
# -----------------------
st.sidebar.header("Export & raw data")
if st.sidebar.button("Download processed CSV"):
    processed = df.copy()
    processed['Date'] = processed['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    csv = processed.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", data=csv, file_name="btc_processed.csv", mime="text/csv")

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw / Processed Data")
    st.dataframe(df.head(200))

st.markdown("---")
st.markdown("Built with ❤️ — upload your CSV or fetch BTC from yfinance. If your CSV has column name issues (e.g. `Date` missing), the app tries to auto-detect common names; otherwise rename your date column to `Date`.")
