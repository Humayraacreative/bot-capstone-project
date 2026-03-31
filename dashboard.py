import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3

# --- MWANZO WA DASHBOARD ---
st.set_page_config(page_title="BOT AI Assistant", layout="wide")

st.title("🏦 BOT Financial Intelligence Dashboard")
st.markdown("### *AI-Driven Analysis & Anomaly Detection*")

# 1. ANZA KWA KUSOMA DATA KUTOKA WAREHOUSE YAKO (SQL Database)
@st.cache_data # Hii inasaidia kuhifadhi data ili isiwez tena
def load_data():
    conn = sqlite3.connect('BOT_Data_Warehouse.db')
    # Tunachagua data yote kutoka table iliyopo
    df = pd.read_sql("SELECT * FROM transactions_warehouse", conn)
    conn.close()
    return df

df = load_data()

# --- SEHEMU YA 2: METRICS (NAMBA ZA MUHIMU SANA) ---
st.subheader("📊 Summary (Muhtasari)")
col1, col2, col3 = st.columns(3)

# Kalkula namba
total_volume = df['Amount'].sum()
suspicious_count = len(df[df['Status'] == 'Suspicious'])
total_tx = len(df)

# Onyesha namba kwenye vibandiko vya rangi
col1.metric("Jumla ya Pesa (TZS)", f"{total_volume:,.0f}")
col2.metric("Miamala Yote", total_tx)
col3.metric("⚠️ Ya Kushangaza (AI)", suspicious_count, delta_color="inverse")

# --- SEHEMU YA 3: CHARTS (MICHO) ---
st.subheader("📉 Mwenendo wa Miamala (Trend)")

# Chagua aina ya miamala kwa ajili ya chati
option = st.selectbox('Chagua Chanzo Cha Kuona Mwenendo:', df['Source'].unique())

# Chuja data kwa hiyo benki/chanzo
df_filtered = df[df['Source'] == option]

# Tengeneza chati
fig = px.line(df_filtered, x='Date', y='Amount', title=f"Mwenendo wa Pesa za {option}", markers=True)
st.plotly_chart(fig, use_container_width=True)

# --- SEHEMU YA 4: DATA YOTE (TABLE) ---
st.subheader("🧾 Orodha ya Miamala ya Kushangaza (Top 10)")

# Chuja tu zile za Suspicious
suspicious_df = df[df['Status'] == 'Suspicious'].sort_values(by='Amount', ascending=False).head(10)

# Onyesha meza yenye rangi
st.dataframe(suspicious_df, use_container_width=True)

# Onyesha orodha kamili (ili wasikie data ni nyingi)
with st.expander("👁️ Bonyeza hapa kuona Data Yote"):
    st.dataframe(df)