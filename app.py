import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
import time

st.set_page_config(page_title="Fraud KPI Dashboard", layout="wide")

st.title("Fraud Analytics Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("fraud_dataset.csv")

    required_cols = ["Time","City","Amount","Fraud","Age"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    # FIX Amount_Balance error
    if "Amount_Balance" not in df.columns:
        df["Amount_Balance"] = df["Amount"].cumsum()

    return df


df = load_data()

# -------------------------------------------------
# MONEY FORMAT
# -------------------------------------------------

def format_money(num):

    if num >= 10000000:
        return f"{round(num/10000000,2)} Cr"

    elif num >= 100000:
        return f"{round(num/100000,2)} Lakh"

    elif num >= 1000:
        return f"{round(num/1000,2)} Thousand"

    else:
        return str(round(num,2))


# -------------------------------------------------
# SIDEBAR SEARCH FILTER
# -------------------------------------------------

st.sidebar.header("Search Filter")

search_city = st.sidebar.text_input("Search City")

city_options = df["City"].unique()

if search_city:
    city_options = [c for c in city_options if search_city.lower() in c.lower()]

select_all = st.sidebar.checkbox("Select All Cities", True)

if select_all:
    city_filter = city_options
else:
    city_filter = st.sidebar.multiselect("Select City", city_options)


age_min = int(df["Age"].min())
age_max = int(df["Age"].max())

age_filter = st.sidebar.slider(
    "Select Age Range",
    age_min,
    age_max,
    (age_min, age_max)
)


filtered_df = df[
    (df["City"].isin(city_filter)) &
    (df["Age"] >= age_filter[0]) &
    (df["Age"] <= age_filter[1])
].copy()

# -------------------------------------------------
# KPI DASHBOARD
# -------------------------------------------------

st.subheader("KPI Dashboard")

total_txn = len(filtered_df)

fraud_cases = filtered_df["Fraud"].sum()

fraud_rate = (fraud_cases / total_txn) * 100 if total_txn > 0 else 0

fraud_amount = filtered_df[filtered_df["Fraud"]==1]["Amount"].sum()

avg_monthly = fraud_amount / 12
future_projection = avg_monthly * 6


col1,col2,col3,col4,col5 = st.columns(5)

col1.metric("Transactions", total_txn)
col2.metric("Fraud Cases", fraud_cases)
col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
col4.metric("Fraud Amount", format_money(fraud_amount))
col5.metric("Next 6 Month Fraud Amount", format_money(future_projection))


st.info(f"""
If the current fraud trend continues, approximately **{format_money(future_projection)}**
fraud amount may occur in the next **6 months**.
""")

st.divider()

# -------------------------------------------------
# AGE ANALYSIS
# -------------------------------------------------

st.header("Age Wise Fraud Analysis")

age_current = filtered_df.groupby("Age")["Fraud"].sum().reset_index()

fig_age = px.bar(
    age_current,
    x="Age",
    y="Fraud",
    title="Current Fraud by Age"
)

st.plotly_chart(fig_age,use_container_width=True)

# AUTO SUMMARY
max_age = age_current.loc[age_current["Fraud"].idxmax()]["Age"]
min_age = age_current.loc[age_current["Fraud"].idxmin()]["Age"]
avg_val = age_current["Fraud"].mean()

st.write(f"""
**Chart Summary**

• Age **{max_age}** group shows **higher fraud activity**  
• Age **{min_age}** group shows **lower fraud activity**  
• Average fraud level across ages is **{round(avg_val,2)}**
""")

# -------------------------------------------------
# AGE FUTURE PREDICTION
# -------------------------------------------------

st.subheader("📈 Predict Age Fraud Trend")

months = st.slider("Select Upcoming Months",1,12,6)

if st.button("Predict Age Fraud Increase"):

    growth_rate = 0.12

    age_future = age_current.copy()

    age_future["FutureFraud"] = (
        age_future["Fraud"] * (1 + growth_rate) ** (months/12)
    )

    fig_future_age = px.line(
        age_future,
        x="Age",
        y="FutureFraud",
        markers=True,
        title=f"Projected Fraud per Age in {months} Months"
    )

    st.plotly_chart(fig_future_age,use_container_width=True)

    st.success(f"""
Prediction indicates potential fraud increase across several age groups
within the next **{months} months** if the trend continues.
""")

st.divider()

# -------------------------------------------------
# TIME ANALYSIS
# -------------------------------------------------

st.header("Time Wise Fraud Analysis")

time_current = filtered_df.groupby("Time")["Fraud"].sum().reset_index()

fig_time = px.line(
    time_current,
    x="Time",
    y="Fraud",
    markers=True,
    title="Current Fraud by Time"
)

st.plotly_chart(fig_time,use_container_width=True)

max_time = time_current.loc[time_current["Fraud"].idxmax()]["Time"]

st.write(f"""
**Chart Summary**

Time **{max_time}** shows the **highest fraud activity**.
Security monitoring should be stronger during this period.
""")

# -------------------------------------------------
# TIME FUTURE
# -------------------------------------------------

st.subheader("Predict Time Fraud Trend")

if st.button("Predict Time Fraud Trend"):

    growth = 0.10

    time_future = time_current.copy()

    time_future["FutureFraud"] = time_future["Fraud"] * (1+growth)

    fig_future_time = px.bar(
        time_future,
        x="Time",
        y="FutureFraud",
        title="Future Fraud Trend by Time"
    )

    st.plotly_chart(fig_future_time,use_container_width=True)

    st.warning("Fraud probability may increase during peak time slots.")

st.divider()

# -------------------------------------------------
# CITY ANALYSIS
# -------------------------------------------------

st.header("City Wise Fraud Analysis")

city_current = filtered_df.groupby("City")["Fraud"].sum().reset_index()

fig_city = px.bar(
    city_current,
    x="City",
    y="Fraud",
    title="Current Fraud by City"
)

st.plotly_chart(fig_city,use_container_width=True)

top_city = city_current.sort_values("Fraud",ascending=False).iloc[0]["City"]

st.write(f"""
**Chart Summary**

City **{top_city}** currently has the **highest fraud cases**.
Authorities should increase fraud monitoring in this location.
""")

# -------------------------------------------------
# CITY RISK SCORE
# -------------------------------------------------

city_current["RiskScore"] = (
    city_current["Fraud"] / city_current["Fraud"].max()
) * 100

fig_risk = px.scatter(
    city_current,
    x="City",
    y="RiskScore",
    size="RiskScore",
    title="Fraud Risk Score per City"
)

st.plotly_chart(fig_risk,use_container_width=True)

st.divider()

# -------------------------------------------------
# CITY FUTURE PREDICTION
# -------------------------------------------------

st.subheader("Predict City Fraud Increase")

if st.button("Predict City Fraud Increase"):

    growth = 0.18

    city_future = city_current.copy()

    city_future["FutureFraud"] = city_future["Fraud"] * (1+growth)

    fig_city_future = px.line(
        city_future,
        x="City",
        y="FutureFraud",
        markers=True,
        title="Future Fraud Increase by City"
    )

    st.plotly_chart(fig_city_future,use_container_width=True)

    future_top = city_future.sort_values("FutureFraud",ascending=False).iloc[0]["City"]

    st.error(f"⚠ City expected highest fraud increase: {future_top}")

st.divider()

# -------------------------------------------------
# AI ANOMALY DETECTION
# -------------------------------------------------

st.header("AI Anomaly Detection")

features = filtered_df[["Amount","Age"]]

model = IsolationForest(contamination=0.05, random_state=42)

filtered_df["Anomaly"] = model.fit_predict(features)

fig_anomaly = px.scatter(
    filtered_df,
    x="Amount",
    y="Age",
    color="Anomaly",
    title="AI Detected Suspicious Transactions"
)

st.plotly_chart(fig_anomaly,use_container_width=True)

st.info("AI model detected unusual transactions which may indicate fraud patterns.")

st.divider()

# -------------------------------------------------
# LIVE FRAUD SIMULATION
# -------------------------------------------------

st.header("Live Fraud Simulation")

if st.button("Start Live Simulation"):

    placeholder = st.empty()

    for i in range(10):

        sample = filtered_df.sample(1)

        city = sample["City"].values[0]
        amount = sample["Amount"].values[0]
        fraud = sample["Fraud"].values[0]

        placeholder.metric(
            label=f"Transaction from {city}",
            value=f"₹ {amount}",
            delta="Fraud Alert!" if fraud==1 else "Normal"
        )

        time.sleep(1)

st.divider()

# -------------------------------------------------
# GLOBAL TREND
# -------------------------------------------------

st.header("Global Fraud Trend")

global_trend = filtered_df.groupby("Time")["Fraud"].sum().reset_index()

fig_global = px.area(
    global_trend,
    x="Time",
    y="Fraud",
    title="Global Fraud Trend Over Time"
)

st.plotly_chart(fig_global,use_container_width=True)

st.divider()

# -------------------------------------------------
# DOWNLOAD DATA
# -------------------------------------------------

st.header("⬇ Download Filtered Data")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="fraud_analysis.csv",
    mime="text/csv"
)
