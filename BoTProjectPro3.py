import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression # HII NI MPYA (KWA UTABIRI)

# ==============================================================================
# SEHEMU YA 1: KUTENGENEZA DATA (Synthetic Data)
# ==============================================================================

print("=== ANZA KWA KUTENGENEZA DATA ===")
data_a = {
    'Date': pd.date_range(start='2024-01-01', periods=100),
    'Transaction_Type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], 100),
    'Amount': np.random.randint(10000, 5000000, 100),
    'Bank': 'CRDB'
}
df_a = pd.DataFrame(data_a)
df_a.to_csv('bank_a_data.csv', index=False)

data_b = {
    'date': pd.date_range(start='2024-03-11', periods=80),
    'Trans_Type': np.random.choice(['deposit', 'withdrawal', 'transfer'], 80), 
    'Amount': np.random.randint(5000, 2000000, 80),
    'Bank': 'nmb'
}
df_b = pd.DataFrame(data_b)
df_b.loc[10:15, 'Amount'] = np.nan 
df_b.to_csv('bank_b_data.csv', index=False)

data_c = {
    'Transaction_Date': pd.date_range(start='2024-01-10', periods=120),
    'Type': np.random.choice(['Payment', 'Withdraw', 'Cash In'], 120),
    'Value': np.random.randint(1000, 1000000, 120),
    'Source': 'M-PESA'
}
df_c = pd.DataFrame(data_c)
df_c.to_csv('agent_data.csv', index=False)

print("✅ Data zimepatikana\n")

# ==============================================================================
# SEHEMU YA 2: FUNCTION SMART
# ==============================================================================

def find_column_name(columns, keywords):
    sorted_cols = sorted(columns, key=len, reverse=True) 
    for col in sorted_cols:
        col_lower = col.lower()
        if 'date' in col_lower or 'time' in col_lower:
            continue
        if any(k in col_lower for k in keywords):
            return col
    return None 

# ==============================================================================
# SEHEMU YA 3: KUSAFISHA NA KUUNGANISHA 
# ==============================================================================

files = glob.glob('*.csv')
all_clean_data = []

print("=== ANZA KUSAFISHA DATA ===")
for file in files:
    # Epuka faili la matokeo yenyewe
    if 'final' in file.lower() or 'cleaned' in file.lower(): 
        continue
    
    print(f"✅ Inasafisha: {file}...")
    df = pd.read_csv(file)
    
    # --- TAFUTA COLUMS ---
    
    # 1. Tafuta Tarehe (Manual Search kuepuka NaT)
    found_date = False  # <--- HII NDIO ILIKUWA INAKOSEKA (Imeongezwa)
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            col_date = col
            found_date = True
            break
    
    if not found_date:  # <--- (Ilirekebishwa kuwa found_date)
        col_date = None
    
    col_amount = find_column_name(df.columns, ['amount', 'value', 'money'])
    col_type = find_column_name(df.columns, ['type', 'category'])
    col_source = find_column_name(df.columns, ['source', 'bank', 'provider'])
    
    # 2. JENGA DATAFRAME SAFI
    df_clean = pd.DataFrame({
        'Date': df[col_date] if col_date else np.nan,
        'Transaction_Type': df[col_type] if col_type else 'Unknown',
        'Amount': df[col_amount] if col_amount else 0,
        'Source': df[col_source] if col_source else 'Unknown'
    })
    
    # 3. KUSAFISHA DATA
    df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce').fillna(0).astype(int)
    df_clean['Transaction_Type'] = df_clean['Transaction_Type'].astype(str).str.capitalize()
    df_clean['Source'] = df_clean['Source'].astype(str).str.upper()
    
    all_clean_data.append(df_clean)

# 4. KUUNGANISHA
final_data = pd.concat(all_clean_data, ignore_index=True)
final_data['Date'] = pd.to_datetime(final_data['Date'], errors='coerce')
print("✅ Data Imesafishwa.\n")

# ==============================================================================
# SEHEMU YA 4A: AI - ANOMALY DETECTION (USALAMA)
# ==============================================================================

print("=== AI ASSISTANT 1: ANOMALY DETECTION (USALAMA) ===")
model_anomaly = IsolationForest(contamination=0.05, random_state=42)
final_data['Anomaly_Score'] = model_anomaly.fit_predict(final_data[['Amount']])
final_data['Status'] = final_data['Anomaly_Score'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')

suspicious_count = len(final_data[final_data['Status'] == 'Suspicious'])
print(f"⚠️  AI Imegundua miamala {suspicious_count} ya Kushangaza.\n")

# ==============================================================================
# SEHEMU YA 4B: AI - PREDICTIVE ANALYTICS (UTABIRI WA MWELEKEO)
# ==============================================================================

print("=== AI ASSISTANT 2: TREND PREDICTION (UTABIRI) ===")

# 1. Weka data kulingana na Tarehe (Jumla ya pesa kila siku)
daily_data = final_data.groupby('Date')['Amount'].sum().reset_index()
daily_data = daily_data.sort_values('Date')

# 2. Tengeneza Feature ya 'Siku ya Mwaka' kwa ajili ya model
daily_data['Day_Number'] = range(len(daily_data))

# 3. Fundisha Model (Linear Regression)
X = daily_data[['Day_Number']]
y = daily_data['Amount']

model_predict = LinearRegression()
model_predict.fit(X, y)

# 4. Tabiri KESHO (Future Value)
next_day_number = len(daily_data) # Siku ya baada ya data iliyopo
predicted_amount_next_day = model_predict.predict([[next_day_number]])

print(f"📊 AI Inatabiri Kesho ({daily_data['Date'].max() + pd.Timedelta(days=1)}):")
print(f"   Jumla ya miamala ya pesa itakuwa karibu: {int(predicted_amount_next_day[0]):,} TZS\n")

# ==============================================================================
# SEHEMU YA 4C: AI - DESCRIPTIVE INSIGHTS (MUHTASARI)
# ==============================================================================

print("=== AI ASSISTANT 3: STRATEGIC SUMMARY (MAELEZO) ===")

# Chanzo kinalipa zaidi?
top_source = final_data.groupby('Source')['Amount'].sum().idxmax()
total_volume = final_data['Amount'].sum()

print(f"🏆 Chanzo kikubwa cha miamala ni: {top_source}")
print(f"💰 Jumla ya kiasi cha pesa kilichohamishwa ni: {total_volume:,} TZS")

# ==============================================================================
# MATOKEO YA MWISHO
# ==============================================================================

print("\n" + "="*50)
print("🎉 **AI ASSISTANT REPORT COMPLETED** 🎉")
print("="*50)
print(f"1. USALAMA: Imegundua {suspicious_count} miamala ya hatari.")
print(f"2. UTABIRI: Kesho inatarajiwa kutoa {int(predicted_amount_next_day[0]):,} TZS.")
print(f"3. MKAKATI: {top_source} inaongoza kwa shughuli.")
print("="*50)

import sqlite3  # Hii library inakuja na Python, usisahau kuitumia

# --- HII NI SEHEMU YA KUJENGA WAREHOUSE (DATABASE) ---

# 1. Ungana na Database (Itaundwa kiotomatiki ikiwa haipo)
conn = sqlite3.connect('BOT_Data_Warehouse.db') 

# 2. Tumpe data yako safi (final_data) kuiweka kwenye SQL
# Tunaita table 'transactions_warehouse'
final_data.to_sql('transactions_warehouse', conn, if_exists='replace', index=False)

print("✅ DATA IMESALIWA KATIKA WAREHOUSE (SQL Database) INSTEAD OF CSV.")
print("✅ Hii ina maana data yako iko tayari kwa ajili ya Big Data Queries.")

# 3. Funga connection
conn.close()