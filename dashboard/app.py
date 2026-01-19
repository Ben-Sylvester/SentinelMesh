import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path

TRACE_FILE = Path("traces.json")
BANDIT_DB = Path("bandit.db")
BUDGET_DB = Path("budget.db")

st.set_page_config(page_title="AI Orchestrator Dashboard", layout="wide")
st.title("🧠 Multi-Model Orchestrator Dashboard")

# ------------------------
# Load Data
# ------------------------

def load_traces():
    if not TRACE_FILE.exists():
        return pd.DataFrame()
    with open(TRACE_FILE) as f:
        return pd.DataFrame(json.load(f))

def load_bandit():
    if not BANDIT_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BANDIT_DB)
    df = pd.read_sql("SELECT * FROM arms", conn)
    conn.close()
    return df

def load_budget():
    if not BUDGET_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BUDGET_DB)
    df = pd.read_sql("SELECT * FROM daily_spend", conn)
    conn.close()
    return df

traces = load_traces()
bandit = load_bandit()
budget = load_budget()

# ------------------------
# KPIs
# ------------------------

st.subheader("📈 System KPIs")

col1, col2, col3, col4 = st.columns(4)

if not traces.empty:
    col1.metric("Total Requests", len(traces))
    col2.metric("Avg Cost ($)", round(traces["cost_usd"].mean(), 4))
    col3.metric("Avg Latency (ms)", int(traces["latency_ms"].mean()))
    col4.metric("Avg Confidence", round(traces["confidence"].mean(), 3))
else:
    st.info("No traces yet. Run some requests first.")

# ------------------------
# Strategy Usage
# ------------------------

st.subheader("🎯 Strategy Usage")

if not traces.empty:
    strategy_counts = traces["strategy"].value_counts()
    st.bar_chart(strategy_counts)

# ------------------------
# Cost Over Time
# ------------------------

st.subheader("💰 Cost Over Time")

if not traces.empty:
    traces["timestamp"] = pd.to_datetime(traces["timestamp"])
    cost_ts = traces.groupby(traces["timestamp"].dt.floor("min"))["cost_usd"].sum()
    st.line_chart(cost_ts)

# ------------------------
# Latency Distribution
# ------------------------

st.subheader("⏱️ Latency Distribution")

if not traces.empty:
    st.line_chart(traces["latency_ms"])

# ------------------------
# Bandit Learning
# ------------------------

st.subheader("🧠 Bandit Learning State")

if not bandit.empty:
    bandit["avg_reward"] = bandit["reward"] / bandit["pulls"].clip(lower=1)
    st.dataframe(bandit)

# ------------------------
# Budget Tracking
# ------------------------

st.subheader("📊 Budget Tracking")

if not budget.empty:
    st.dataframe(budget)

# ------------------------
# Raw Traces
# ------------------------

st.subheader("📜 Recent Traces")

if not traces.empty:
    st.dataframe(traces.tail(25))
