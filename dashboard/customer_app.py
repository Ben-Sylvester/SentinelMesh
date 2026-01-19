import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path

TENANT_DB = Path("tenants.db")
BUDGET_DB = Path("tenant_budget.db")
TRACE_FILE = Path("traces.json")

st.set_page_config(page_title="Customer AI Dashboard", layout="wide")
st.title("👤 Customer AI Usage Dashboard")

# ----------------------------
# Helpers
# ----------------------------

def get_tenant_by_key(api_key: str):
    conn = sqlite3.connect(TENANT_DB)
    row = conn.execute("""
        SELECT id, name, daily_limit, requests_per_minute
        FROM tenants WHERE api_key = ?
    """, (api_key,)).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "name": row[1],
        "daily_limit": row[2],
        "rpm": row[3],
    }


def load_spend(tenant_id: str):
    conn = sqlite3.connect(BUDGET_DB)
    df = pd.read_sql("""
        SELECT * FROM spend WHERE tenant_id = ?
        ORDER BY day DESC
    """, conn, params=(tenant_id,))
    conn.close()
    return df


def load_traces(tenant_id: str):
    if not TRACE_FILE.exists():
        return pd.DataFrame()

    with open(TRACE_FILE) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        return df

    return df[df.get("tenant_id") == tenant_id]


# ----------------------------
# Login
# ----------------------------

st.sidebar.subheader("🔐 API Key Login")

api_key = st.sidebar.text_input("Enter your API key", type="password")

if not api_key:
    st.info("Enter your API key to view your dashboard.")
    st.stop()

tenant = get_tenant_by_key(api_key)

if not tenant:
    st.error("Invalid API key.")
    st.stop()

# ----------------------------
# Tenant Header
# ----------------------------

st.success(f"Welcome: **{tenant['name']}**")

# ----------------------------
# Load Data
# ----------------------------

spend_df = load_spend(tenant["id"])
traces_df = load_traces(tenant["id"])

today_spend = float(spend_df.iloc[0]["total"]) if not spend_df.empty else 0.0
remaining = max(tenant["daily_limit"] - today_spend, 0)

# ----------------------------
# KPIs
# ----------------------------

st.subheader("📊 Account Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Daily Limit ($)", tenant["daily_limit"])
col2.metric("Today's Spend ($)", round(today_spend, 3))
col3.metric("Remaining ($)", round(remaining, 3))
col4.metric("Rate Limit (RPM)", tenant["rpm"])

# ----------------------------
# Usage Metrics
# ----------------------------

if not traces_df.empty:
    st.subheader("⚡ Usage Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Requests", len(traces_df))
    col2.metric("Avg Latency (ms)", int(traces_df["latency_ms"].mean()))
    col3.metric("Avg Cost ($)", round(traces_df["cost_usd"].mean(), 4))

# ----------------------------
# Strategy Usage
# ----------------------------

if not traces_df.empty:
    st.subheader("🎯 Strategy Usage")
    st.bar_chart(traces_df["strategy"].value_counts())

# ----------------------------
# Cost Over Time
# ----------------------------

if not traces_df.empty:
    st.subheader("💰 Cost Over Time")
    traces_df["timestamp"] = pd.to_datetime(traces_df["timestamp"])
    cost_ts = traces_df.groupby(traces_df["timestamp"].dt.floor("min"))["cost_usd"].sum()
    st.line_chart(cost_ts)

# ----------------------------
# Recent Requests
# ----------------------------

st.subheader("📜 Recent Requests")

if not traces_df.empty:
    st.dataframe(
        traces_df.sort_values("timestamp", ascending=False)
        .head(20)[
            ["timestamp", "strategy", "cost_usd", "latency_ms", "confidence", "reason"]
        ]
    )
else:
    st.info("No requests yet for this tenant.")

# ----------------------------
# Refresh
# ----------------------------

st.sidebar.button("🔄 Refresh Page", on_click=lambda: st.experimental_rerun())
