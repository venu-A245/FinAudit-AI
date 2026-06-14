# FinAudit-AI

Agentic Graph-RAG financial audit demo built with LangGraph, synthetic data, and local Ollama.

## What This Demo Shows

- Multi-source retrieval across:
	- transaction records
	- audit logs
	- financial statements
- LangGraph orchestration with source routing and retrieval/synthesis nodes
- Graph-RAG style retrieval with entity-link expansion for stronger evidence recall
- Source-cited responses in format `[source_type:doc_id]`
- Streamlit UI for interactive hands-on exploration

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── scripts/
│   └── generate_synthetic_data.py
├── data/
│   └── synthetic/
└── src/
		└── finaudit/
				├── __init__.py
				├── agent.py
				├── config.py
				├── data_loader.py
				└── retrieval.py
```

## Prerequisites

- Python 3.10+
- Ollama installed and running locally

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Pull a local model (example):

```bash
ollama pull llama3.1:8b
```

3. Generate synthetic data:

```bash
python scripts/generate_synthetic_data.py --out data/synthetic --transactions 1000 --audit-logs 180 --months 12
```

4. Start the demo:

```bash
streamlit run app.py
```

## Demo Query Ideas

- "Show high-risk transaction patterns linked to vendor anomalies this quarter."
- "Correlate critical audit logs with related transactions and suggest next controls."
- "Summarize margin pressure signs and potential fraud indicators from statements and logs."

## Environment Variables

- `FINAUDIT_DATA_DIR` (default: `data/synthetic`)
- `FINAUDIT_OLLAMA_MODEL` (default: `llama3.1:8b`)
- `FINAUDIT_TOP_K` (default: `4`)
- `FINAUDIT_GRAPH_HOP_BOOST` (default: `0.12`)

## Notes

- Data is synthetic and for demo/training purposes only.
- Retrieval uses a lightweight hybrid strategy (lexical + graph-neighbor boost).
- The app shows retrieved evidence and metadata to keep audit reasoning traceable.
