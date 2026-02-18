"""
SentinelMesh — Admin Dashboard

FIX: Previous version read from traces.json, bandit.db, and budget.db —
all of which were replaced by the unified SQLite database (data/learning_state.db)
and the tenant_budget.db. Dashboard was showing empty data on every launch.
Now reads directly from the live databases.
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path

DB_PATH     = Path("data/learning_state.db")
BUDGET_DB   = Path("tenant_budget.db")
TENANT_DB   = Path("tenants.db")

st.set_page_config(page_title="SentinelMesh Admin Dashboard", layout="wide")
st.title("🧠 SentinelMesh — Admin Dashboard")

# ── Data Loaders ────────────────────────────────────────────────────────────

def load_traces(limit: int = 500) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT id, timestamp, strategy, data FROM traces ORDER BY id DESC LIMIT {limit}",
        conn,
    )
    conn.close()
    if df.empty:
        return df
    # Expand JSON data column into individual columns
    expanded = df["data"].apply(lambda x: json.loads(x) if x else {})
    meta     = pd.json_normalize(expanded)
    return pd.concat([df[["id", "timestamp", "strategy"]], meta], axis=1)


def load_bandit() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT arm, data FROM bandit_state", conn)
    conn.close()
    if df.empty:
        return df
    expanded = df["data"].apply(lambda x: json.loads(x) if x else {})
    meta     = pd.json_normalize(expanded)
    return pd.concat([df["arm"], meta], axis=1)


def load_rl_stats() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT state, action, value, count FROM rl_qvalues ORDER BY value DESC LIMIT 100", conn)
    conn.close()
    return df


def load_budget() -> pd.DataFrame:
    if not BUDGET_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BUDGET_DB)
    df   = pd.read_sql("SELECT tenant_id, day, total FROM spend ORDER BY day DESC", conn)
    conn.close()
    return df


def load_tenants() -> pd.DataFrame:
    if not TENANT_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(TENANT_DB)
    df   = pd.read_sql("SELECT id, name, daily_limit, requests_per_minute FROM tenants", conn)
    conn.close()
    return df

# ── Load ────────────────────────────────────────────────────────────────────

traces  = load_traces()
bandit  = load_bandit()
rl_data = load_rl_stats()
budget  = load_budget()
tenants = load_tenants()

# ── KPIs ────────────────────────────────────────────────────────────────────

st.subheader("📈 System KPIs")
col1, col2, col3, col4 = st.columns(4)

if not traces.empty:
    col1.metric("Total Requests",    len(traces))
    col2.metric("Avg Cost ($)",      round(traces["cost_usd"].mean(), 4)  if "cost_usd"    in traces.columns else "N/A")
    col3.metric("Avg Latency (ms)",  int(traces["latency_ms"].mean())      if "latency_ms"  in traces.columns else "N/A")
    col4.metric("Avg Confidence",    round(traces["confidence"].mean(), 3) if "confidence"  in traces.columns else "N/A")
else:
    st.info("No traces yet. Run some requests first.")

# ── Strategy Usage ───────────────────────────────────────────────────────────

st.subheader("🎯 Strategy Usage")
if not traces.empty and "strategy" in traces.columns:
    st.bar_chart(traces["strategy"].value_counts())

# ── Cost Over Time ───────────────────────────────────────────────────────────

st.subheader("💰 Cost Over Time")
if not traces.empty and "timestamp" in traces.columns and "cost_usd" in traces.columns:
    traces["timestamp"] = pd.to_datetime(traces["timestamp"], errors="coerce")
    cost_ts = traces.groupby(traces["timestamp"].dt.floor("min"))["cost_usd"].sum()
    st.line_chart(cost_ts)

# ── Bandit Learning State ─────────────────────────────────────────────────────

st.subheader("🎰 Bandit (LinUCB) Arms")
if not bandit.empty:
    st.dataframe(bandit)

# ── RL Q-Values ───────────────────────────────────────────────────────────────

st.subheader("🤖 RL Q-Values (Top 100)")
if not rl_data.empty:
    st.dataframe(rl_data)

# ── Budget Tracking ───────────────────────────────────────────────────────────

st.subheader("📊 Tenant Budget Usage")
if not budget.empty:
    st.dataframe(budget)

# ── Tenants ───────────────────────────────────────────────────────────────────

st.subheader("👥 Registered Tenants")
if not tenants.empty:
    st.dataframe(tenants)

# ── Raw Traces ────────────────────────────────────────────────────────────────

st.subheader("📜 Recent Traces (last 25)")
if not traces.empty:
    st.dataframe(traces.head(25))

st.sidebar.button("🔄 Refresh", on_click=st.rerun)
