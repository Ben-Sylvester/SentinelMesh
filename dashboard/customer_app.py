"""
SentinelMesh — Customer Dashboard

FIX: Previous version read trace data from traces.json (deleted in refactor).
Now reads from the SQLite traces table in data/learning_state.db, filtered
by tenant_id stored in each trace's JSON data column.
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path

DB_PATH   = Path("data/learning_state.db")
BUDGET_DB = Path("tenant_budget.db")
TENANT_DB = Path("tenants.db")

st.set_page_config(page_title="Customer AI Dashboard", layout="wide")
st.title("👤 Customer AI Usage Dashboard")

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_tenant_by_key(api_key: str):
    if not TENANT_DB.exists():
        return None
    conn = sqlite3.connect(TENANT_DB)
    row  = conn.execute(
        "SELECT id, name, daily_limit, requests_per_minute FROM tenants WHERE api_key = ?",
        (api_key,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "daily_limit": row[2], "rpm": row[3]}


def load_spend(tenant_id: str) -> pd.DataFrame:
    if not BUDGET_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BUDGET_DB)
    df   = pd.read_sql(
        "SELECT day, total FROM spend WHERE tenant_id = ? ORDER BY day DESC",
        conn,
        params=(tenant_id,),
    )
    conn.close()
    return df


def load_traces(tenant_id: str, limit: int = 200) -> pd.DataFrame:
    """
    FIX: reads from SQLite traces table, not deleted traces.json.
    Filters by tenant_id stored inside each trace's JSON data blob.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"SELECT data FROM traces ORDER BY id DESC LIMIT {limit}"
    ).fetchall()
    conn.close()

    records = []
    for (raw,) in rows:
        try:
            rec = json.loads(raw)
            # Filter to this tenant (stored in features.tenant_id)
            features = rec.get("features", {})
            if features.get("tenant_id") == tenant_id:
                records.append(rec)
        except Exception:
            pass

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Login ─────────────────────────────────────────────────────────────────────

st.sidebar.subheader("🔐 API Key Login")
api_key = st.sidebar.text_input("Enter your API key", type="password")

if not api_key:
    st.info("Enter your API key to view your dashboard.")
    st.stop()

tenant = get_tenant_by_key(api_key)
if not tenant:
    st.error("Invalid API key.")
    st.stop()

st.success(f"Welcome: **{tenant['name']}**")

# ── Load Data ─────────────────────────────────────────────────────────────────

spend_df  = load_spend(tenant["id"])
traces_df = load_traces(tenant["id"])

today_spend = float(spend_df.iloc[0]["total"]) if not spend_df.empty else 0.0
remaining   = max(tenant["daily_limit"] - today_spend, 0.0)

# ── KPIs ──────────────────────────────────────────────────────────────────────

st.subheader("📊 Account Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Daily Limit ($)",  tenant["daily_limit"])
col2.metric("Today's Spend ($)", round(today_spend, 3))
col3.metric("Remaining ($)",    round(remaining, 3))
col4.metric("Rate Limit (RPM)", tenant["rpm"])

# ── Usage Metrics ──────────────────────────────────────────────────────────────

if not traces_df.empty:
    st.subheader("⚡ Usage Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requests", len(traces_df))
    col2.metric("Avg Latency (ms)", int(traces_df["latency_ms"].mean()) if "latency_ms" in traces_df.columns else "N/A")
    col3.metric("Avg Cost ($)",     round(traces_df["cost_usd"].mean(), 4) if "cost_usd" in traces_df.columns else "N/A")

    st.subheader("🎯 Strategy Usage")
    if "strategy" in traces_df.columns:
        st.bar_chart(traces_df["strategy"].value_counts())

    st.subheader("💰 Cost Over Time")
    if "timestamp" in traces_df.columns and "cost_usd" in traces_df.columns:
        traces_df["timestamp"] = pd.to_datetime(traces_df["timestamp"], errors="coerce")
        cost_ts = traces_df.groupby(traces_df["timestamp"].dt.floor("min"))["cost_usd"].sum()
        st.line_chart(cost_ts)

    st.subheader("📜 Recent Requests")
    display_cols = [c for c in ["timestamp", "strategy", "cost_usd", "latency_ms", "confidence", "reason"] if c in traces_df.columns]
    st.dataframe(traces_df[display_cols].head(20))
else:
    st.info("No requests yet for this tenant.")

st.sidebar.button("🔄 Refresh Page", on_click=st.rerun)
