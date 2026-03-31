import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
import sqlite3
import io

# --- CONFIGURATIONS ---
st.set_page_config(page_title="BOT AI Intelligent System", layout="wide")

# --- HELPER FUNCTIONS (LOGIC YA APP) ---

# 1. Function ya Kusafisha Data (Smart Cleaning) - Iliyoko ndani ya App
def clean_pipeline(df_raw):
    # Tunarudisha hiyo function ya tatu ya kutafuta columns
    def find_col(cols, keywords, avoid=None):
        sorted_cols = sorted(cols, key=len, reverse=True)
        for col in sorted_cols:
            col_low = col.lower()
            if avoid:
                if any(a in col_low for a in avoid): continue
            if any(k in col_low for k in keywords): return col
        return None

    # Tafuta Columns
    col_date = find_col(df_raw.columns, ['date', 'time'])
    col_amt = find_col(df_raw.columns, ['amount', 'value', 'money'])
    col_type = find_col(df_raw.columns, ['type', 'category'], avoid=['date'])
    col_src = find_col(df_raw.columns, ['source', 'bank', 'provider'])

    # Jenga Data Safi
    df_clean = pd.DataFrame({
        'Date': pd.to_datetime(df_raw[col_date]) if col_date else pd.NaT,
        'Transaction_Type': df_raw[col_type] if col_type else 'Unknown',
        'Amount': pd.to_numeric(df_raw[col_amt], errors='coerce').fillna(0).astype(int),
        'Source': df_raw[col_src] if col_src else 'Unknown'
    })
    
    # Text Standardization
    df_clean['Transaction_Type'] = df_clean['Transaction_Type'].astype(str).str.capitalize()
    df_clean['Source'] = df_clean['Source'].astype(str).str.upper()
    return df_clean

# 2. Function ya AI Analysis
def run_ai_analysis(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[['Amount']])
    df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')
    return df

# 3. Function ya Prediction (Linear Regression)
def predict_trend(df):
    daily = df.groupby('Date')['Amount'].sum().reset_index().sort_values('Date')
    if len(daily) < 10: return 0 # Data ni kidogo, hawezi tabiri
    
    from sklearn.linear_model import LinearRegression
    X = np.array(range(len(daily))).reshape(-1, 1)
    y = daily['Amount'].values
    model = LinearRegression().fit(X, y)
    next_day_pred = model.predict([[len(daily)]])
    return int(next_day_pred[0])


# --- MAIN APP UI ---

st.title("🏦 BOT Financial Intelligence System")
st.markdown("### *Automated ETL & AI Assistant Platform*")

# --- SIDEBAR (CONTROL PANEL) ---
st.sidebar.header("🔧 System Control")

option = st.sidebar.radio("Chagua Chanzo cha Data:", ["Run Simulation (Demo)", "Upload Your Own CSV"])

df = None

if option == "Run Simulation (Demo)":
    st.sidebar.info("Inaendesha Pipeline ya Data ya mfano...")
    
    # 1. Generate Data Iliyochafu (Simulation)
    if st.sidebar.button("🚀 RUN FULL PIPELINE"):
        with st.spinner('Inasafisha Data, Kujaza Warehouse, na Kufanya AI Analysis...'):
            # Generate synthetic data
            dates = pd.date_range('2024-01-01', periods=100)
            data = {
                'Transaction_Date': dates,
                'Type': np.random.choice(['Deposit', 'Withdraw', 'Transfer'], 100),
                'Value': np.random.randint(10000, 5000000, 100),
                'Bank': np.random.choice(['CRDB', 'NMB', 'M-PESA'], 100)
            }
            df_raw = pd.DataFrame(data)
            
            # Run Pipeline
            df_clean = clean_pipeline(df_raw)
            df_ai = run_ai_analysis(df_clean)
            pred = predict_trend(df_ai)
            
            # Hifadhi kwenye Session State (Kuonyesha kwenye Dashboard)
            st.session_state['data'] = df_ai
            st.session_state['prediction'] = pred
            st.success("✅ Pipeline Imemaliza! Data imeshapitiwa AI Analysis.")

elif option == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Pakua faili la CSV", type="csv")
    if uploaded_file is not None:
        with st.spinner('Inaprosesha Faili Lako...'):
            df_raw = pd.read_csv(uploaded_file)
            df_clean = clean_pipeline(df_raw)
            df_ai = run_ai_analysis(df_clean)
            pred = predict_trend(df_ai)
            
            st.session_state['data'] = df_ai
            st.session_state['prediction'] = pred
            st.success("✅ Faili Imeproseswa!")


# --- DASHBOARD DISPLAY (HII INAONEKANA TU BAADA YA KUFANYA KAZI) ---
if 'data' in st.session_state:
    df = st.session_state['data']
    pred_val = st.session_state['prediction']

    # --- METRICS ---
    st.subheader("📊 Real-Time Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    total_vol = df['Amount'].sum()
    suspicious = len(df[df['Status'] == 'Suspicious'])
    normal = len(df[df['Status'] == 'Normal'])
    
    col1.metric("💰 Jumla ya Pesa", f"{total_vol:,.0f}")
    col2.metric("📉 Miamala Yote", len(df))
    col3.metric("⚠️ Hatari (AI)", suspicious, delta_color="inverse")
    col4.metric("📈 Utabiri wa Kesho", f"{pred_val:,.0f}")

    # --- CHARTS ---
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Trend ya Miamala (Kila Siku)")
        daily = df.groupby('Date')['Amount'].sum().reset_index()
        fig = px.line(daily, x='Date', y='Amount', title="Volume Kila Siku")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader ("Chanzo vs Status (Usalama)")
        source_status = df.groupby(['Source', 'Status']).count().reset_index()
        fig_bar = px.bar(source_status, x='Source', y='Amount', color='Status', barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("🧾 Miamala Iliyogunduliwa na AI (Suspicious)")
    suspicious_data = df[df['Status'] == 'Suspicious']
    st.dataframe(suspicious_data, use_container_width=True)

else:
    st.info("👈 Tafadhali chagua chanzo cha data kwenye menyu ya kushoto (Sidebar) ili kuanza.")
