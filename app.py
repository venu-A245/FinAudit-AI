from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st

from src.finaudit.agent import build_audit_app
from src.finaudit.config import CONFIG, DemoConfig


st.set_page_config(page_title="FinAudit-AI Demo", page_icon="FA", layout="wide")
st.title("FinAudit-AI: Agentic Graph-RAG Financial Audit Demo")
st.caption("LangGraph workflow + synthetic data + source-cited audit reasoning")


def ensure_data_exists(data_dir: str) -> None:
    needed = ["transactions.csv", "audit_logs.json", "financial_statements.json"]
    missing = [p for p in needed if not Path(data_dir, p).exists()]
    if not missing:
        return

    cmd = [
        "python",
        "scripts/generate_synthetic_data.py",
        "--out",
        data_dir,
        "--transactions",
        "1000",
        "--audit-logs",
        "180",
        "--months",
        "12",
    ]
    subprocess.run(cmd, check=True)


def render_sources(evidence_items: list[dict]) -> None:
    st.subheader("Retrieved Evidence")
    for i, item in enumerate(evidence_items, start=1):
        header = f"{i}. [{item['source_type']}:{item['doc_id']}] score={item['score']}"
        with st.expander(header, expanded=False):
            st.write(item["text"])
            st.json(item["metadata"])


with st.sidebar:
    st.header("Demo Controls")
    model_name = st.text_input("Ollama model", value=CONFIG.model_name)
    top_k = st.slider("Top-K per source", min_value=2, max_value=8, value=CONFIG.top_k_per_source)
    graph_boost = st.slider("Graph hop boost", min_value=0.0, max_value=0.5, value=float(CONFIG.graph_hop_boost), step=0.01)

    if st.button("Regenerate synthetic data"):
        subprocess.run(
            [
                "python",
                "scripts/generate_synthetic_data.py",
                "--out",
                CONFIG.data_dir,
                "--transactions",
                "1000",
                "--audit-logs",
                "180",
                "--months",
                "12",
            ],
            check=True,
        )
        st.success("Synthetic data regenerated.")

os.environ["FINAUDIT_OLLAMA_MODEL"] = model_name
os.environ["FINAUDIT_TOP_K"] = str(top_k)
os.environ["FINAUDIT_GRAPH_HOP_BOOST"] = str(graph_boost)

ensure_data_exists(CONFIG.data_dir)
runtime_config = DemoConfig(
    data_dir=CONFIG.data_dir,
    model_name=model_name,
    top_k_per_source=top_k,
    graph_hop_boost=graph_boost,
)
app = build_audit_app(runtime_config)

sample_prompts = [
    "Show high-risk transaction patterns linked to vendor anomalies this quarter.",
    "Correlate critical audit logs with related transactions and suggest next controls.",
    "Summarize margin pressure signs and potential fraud indicators from statements and logs.",
]

query = st.text_area("Ask an audit question", value=sample_prompts[0], height=120)
if st.button("Run audit analysis", type="primary"):
    with st.spinner("Running LangGraph workflow..."):
        result = app.invoke({"query": query})

    st.subheader("Agent Answer")
    st.markdown(result.get("answer", "No answer generated."))

    retrieved = result.get("retrieved", [])
    if retrieved:
        render_sources(retrieved)
    else:
        st.info("No evidence retrieved.")

st.divider()
st.markdown("### Notes")
st.markdown("- This demo is synthetic and for workflow illustration only.")
st.markdown("- Ensure Ollama is running locally before querying.")
