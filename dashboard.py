import streamlit as st
import requests
import pandas as pd
import threading
import json
import websocket

API = "http://localhost:8000"

st.title("🧠 Orchestrator Intelligence Dashboard")

# -----------------------
# Session State
# -----------------------
if "live_traces" not in st.session_state:
    st.session_state.live_traces = []

if "ws_started" not in st.session_state:
    st.session_state.ws_started = False

# -----------------------
# Metrics: Belief Heatmap
# -----------------------
beliefs = requests.get(f"{API}/metrics/beliefs").json()

rows = []
for signature, strategies in beliefs.items():
    for strategy, stats in strategies.items():
        rows.append({
            "signature": signature,
            "strategy": strategy,
            "avg_reward": stats.get("avg_reward", 0),
            "count": stats.get("count", 0),
        })

df = pd.DataFrame(rows)

if not df.empty:
    pivot = df.pivot("signature", "strategy", "avg_reward")
    st.subheader("🔥 Belief Heatmap (Reward)")
    st.dataframe(pivot, use_container_width=True)

# -----------------------
# Strategy Drift
# -----------------------
drift = requests.get(f"{API}/metrics/strategy-drift").json()

st.subheader("📈 Strategy Confidence Drift")

for strategy, points in drift.items():
    df = pd.DataFrame(points)
    if not df.empty:
        st.line_chart(df["reward"])

# -----------------------
# Model ROI
# -----------------------
roi = requests.get(f"{API}/metrics/roi").json()
st.subheader("💰 Model ROI")

if roi:
    st.dataframe(pd.DataFrame(roi).T, use_container_width=True)

# -----------------------
# WebSocket: Live Traces
# -----------------------
def on_message(ws, message):
    data = json.loads(message)
    if data.get("type") == "trace":
        st.session_state.live_traces.append(data["payload"])

        # cap memory
        st.session_state.live_traces = st.session_state.live_traces[-200:]


def start_ws():
    ws = websocket.WebSocketApp(
        "ws://localhost:8000/ws",
        on_message=on_message
    )
    ws.run_forever()


if not st.session_state.ws_started:
    st.session_state.ws_started = True
    threading.Thread(target=start_ws, daemon=True).start()

# -----------------------
# UI: Live Trace Feed
# -----------------------
st.subheader("⚡ Live Traces")

live_traces = st.session_state.live_traces

if live_traces:
    with st.container(height=300):
        st.json(live_traces[-5:])
else:
    st.info("Waiting for live traffic...")
