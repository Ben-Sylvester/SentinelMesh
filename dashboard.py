"""
SentinelMesh — Live Intelligence Dashboard (root-level entry point)

FIX: Previous version made HTTP requests at module import time:
    beliefs = requests.get(f"{API}/metrics/beliefs").json()
This crashes immediately with ConnectionRefusedError if the API server
isn't running, making the dashboard unusable even in development.

Now all API calls are inside functions and wrapped in try/except,
so the dashboard loads gracefully and shows friendly warnings if
the API is offline.
"""

import streamlit as st
import requests
import pandas as pd
import threading
import json

API = "http://localhost:8000"

st.title("🧠 SentinelMesh — Intelligence Dashboard")


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, default=None):
    """Safe API fetch — returns default on any error instead of crashing."""
    try:
        r = requests.get(f"{API}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.warning(f"API unavailable ({path}): {exc}")
        return default


# ── Session State ─────────────────────────────────────────────────────────────

if "live_traces" not in st.session_state:
    st.session_state.live_traces = []

if "ws_started" not in st.session_state:
    st.session_state.ws_started = False


# ── Belief Heatmap ────────────────────────────────────────────────────────────

st.subheader("🔥 Belief Heatmap (Reward by Signature × Strategy)")

beliefs = api_get("/metrics/beliefs", default={})
if beliefs:
    rows = []
    for signature, strategies in beliefs.items():
        for strategy_key, stats in strategies.items():
            rows.append({
                "signature": signature[:12] + "…",
                "strategy":  strategy_key,
                "avg_reward": stats.get("mean", 0),
                "count":      stats.get("count", 0),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        pivot = df.pivot_table(index="signature", columns="strategy", values="avg_reward", aggfunc="mean")
        st.dataframe(pivot, use_container_width=True)
else:
    st.info("No belief data yet. Run some requests first.")


# ── Strategy Drift ────────────────────────────────────────────────────────────

st.subheader("📈 Strategy Reward Drift")
drift = api_get("/metrics/strategy-drift", default={})
for strategy, points in (drift or {}).items():
    df_d = pd.DataFrame(points)
    if not df_d.empty and "reward" in df_d.columns:
        st.caption(strategy)
        st.line_chart(df_d["reward"])


# ── Model ROI ─────────────────────────────────────────────────────────────────

st.subheader("💰 Model ROI")
roi = api_get("/metrics/roi", default={})
if roi:
    st.dataframe(pd.DataFrame(roi).T, use_container_width=True)


# ── RL Stats ──────────────────────────────────────────────────────────────────

st.subheader("🤖 RL Policy Stats")
rl = api_get("/admin/rl-stats", default={})
if rl:
    st.json(rl)


# ── WebSocket Live Feed ───────────────────────────────────────────────────────

def _on_ws_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("type") == "trace":
            st.session_state.live_traces.append(data["payload"])
            st.session_state.live_traces = st.session_state.live_traces[-200:]
    except Exception:
        pass


def _start_ws():
    try:
        import websocket
        ws = websocket.WebSocketApp(f"ws://localhost:8000/ws", on_message=_on_ws_message)
        ws.run_forever()
    except Exception:
        pass


if not st.session_state.ws_started:
    st.session_state.ws_started = True
    threading.Thread(target=_start_ws, daemon=True).start()

st.subheader("⚡ Live Traces")
if st.session_state.live_traces:
    with st.container(height=300):
        st.json(st.session_state.live_traces[-5:])
else:
    st.info("Waiting for live traffic…")

st.sidebar.button("🔄 Refresh", on_click=st.rerun)
