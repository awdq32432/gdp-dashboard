import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# --- Page Setup ---
st.set_page_config(page_title="Prime Broker Dashboard", layout="wide")
st.title("📊 Prime Broker Client Dashboard")

# --- Sidebar Settings ---
st.sidebar.header("🔧 Settings")
maintenance_margin = st.sidebar.slider(
    "Maintenance Margin %", 0.0, 1.0, 0.2, 0.01,
    help="Percentage of position value required as margin"
)
var_confidence = st.sidebar.slider(
    "VaR Confidence Level", 0.90, 0.99, 0.95, 0.005,
    help="Confidence level for 1-day Historical VaR"
)

# --- Portfolio Input ---
st.markdown("## 1. Portfolio Input")
st.markdown("## 1. Portfolio Input")
if 'portfolio_df' not in st.session_state:
    default = ["CBA.AX","BHP.AX","WES.AX","TLS.AX","ANZ.AX"]
    st.session_state.portfolio_df = pd.DataFrame({
        'Ticker': default,
        'Quantity': [500,400,300,200,100],
        'CostBasis': [50,40,30,20,10]
    })
portfolio_df = st.data_editor(
    st.session_state.portfolio_df,
    num_rows='dynamic',
    use_container_width=True
)
st.session_state.portfolio_df = portfolio_df

# --- Fetch Prices ---
@st.cache_data(ttl=60)
def get_prices(tickers):
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = yf.download(tickers, period='2d', interval='1d', progress=False)['Close']
    return df.iloc[-1], df.iloc[-2]

live_prices, prev_prices = get_prices(portfolio_df['Ticker'].tolist())

# --- Compute Metrics ---
df = portfolio_df.copy()
df['PrevClose'] = df['Ticker'].map(prev_prices)
df['LivePrice'] = df['Ticker'].map(live_prices)
df['DailyPL'] = (df['LivePrice'] - df['PrevClose']) * df['Quantity']
df['DailyPL%'] = (df['LivePrice']/df['PrevClose'] - 1) * 100
df['UnrealizedPL'] = (df['LivePrice'] - df['CostBasis']) * df['Quantity']

# Portfolio Performance
cum_pl = df['UnrealizedPL'].sum().item()
equity_start = (df['CostBasis'] * df['Quantity']).sum()
cum_return = cum_pl / equity_start if equity_start else 0
years = max((datetime.now() - datetime.now()).days / 365, 1)  # placeholder for actual period
cagr = (1 + cum_return) ** (1/years) - 1

# Margin Ratio
used_margin = (df['LivePrice'] * df['Quantity'] * maintenance_margin).sum()
equity_value = df['UnrealizedPL'].sum() + equity_start
margin_ratio = used_margin / equity_value if equity_value else np.nan

# Historical VaR
def historical_var(arr, alpha):
    arr = np.sort(arr)
    idx = alpha * (len(arr)-1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx)); w = idx - lo
    return arr[lo] + w * (arr[hi] - arr[lo])
var95 = historical_var(df['DailyPL%'].fillna(0).values, var_confidence)

# Reconciliation
cp = df[['Ticker','Quantity']].copy()
cp['Price_cp'] = df['LivePrice'] * (1 + np.random.normal(0,0.001,len(df)))
breaks = df.merge(cp,on=['Ticker','Quantity'])
breaks = breaks[np.abs(breaks['LivePrice']-breaks['Price_cp']) > 1e-6]

# --- Display KPIs ---
st.markdown("## 2. Portfolio Performance")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Cumulative P/L", f"{cum_pl:,.2f}", delta=f"{cum_return*100:.2f}%")
col2.metric("CAGR", f"{cagr*100:.2f}%")
col3.metric("Margin Ratio", f"{margin_ratio:.1%}")
col4.metric(f"{int(var_confidence*100)}% VaR", f"{var95:.2f}%")

# --- Data & Charts ---
st.markdown("## 3. Detailed View")
st.dataframe(df, use_container_width=True)
st.markdown("### Charts")
chart1,chart2 = st.columns(2)
with chart1:
    st.bar_chart(df.set_index('Ticker')['DailyPL%'], height=300)
with chart2:
    st.line_chart(df.set_index('Ticker')['UnrealizedPL'], height=300)

# --- Reconciliation ---
st.markdown("## 4. Reconciliation")
st.write(f"Breaks: {len(breaks)}")
st.dataframe(breaks[['Ticker','Quantity','LivePrice','Price_cp']], use_container_width=True)

# --- PDF Export ---
st.markdown("## 5. Export")
def gen_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold",16)
    c.drawString(40,750,"Prime Broker Dashboard Report")
    c.setFont("Helvetica",10)
    c.drawString(40,730,f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    y=700
    for label,value in [
        ("Cumulative P/L", f"{cum_pl:,.2f}"),
        ("CAGR", f"{cagr*100:.2f}%"),
        ("Margin Ratio", f"{margin_ratio:.1%}"),
        (f"{int(var_confidence*100)}% VaR", f"{var95:.2f}%"),
        ("Reconciliation Breaks", str(len(breaks)))
    ]:
        c.drawString(40,y,f"{label}: {value}")
        y -= 15
    c.showPage(); c.save(); buffer.seek(0)
    return buffer.getvalue()

if st.button("Download PDF Report"):
    pdf = gen_pdf()
    st.download_button("Download PDF", pdf, file_name="report.pdf", mime="application/pdf")
