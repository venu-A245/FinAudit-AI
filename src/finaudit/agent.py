from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .config import DemoConfig
from .data_loader import SourceDocument, load_all_sources
from .retrieval import GraphRAGRetriever, RankedDocument


class AuditState(TypedDict, total=False):
    query: str
    selected_sources: list[str]
    retrieved: list[dict[str, Any]]
    answer: str


def route_sources(query: str) -> list[str]:
    q = query.lower()
    selected: list[str] = []

    if any(k in q for k in ["txn", "transaction", "vendor", "invoice", "payment", "fraud"]):
        selected.append("transactions")
    if any(k in q for k in ["audit", "control", "exception", "log", "severity"]):
        selected.append("audit_logs")
    if any(k in q for k in ["margin", "revenue", "expense", "statement", "cash flow", "profit"]):
        selected.append("financial_statements")

    if not selected:
        return ["transactions", "audit_logs", "financial_statements"]
    return selected


def _to_citation(rank: RankedDocument) -> dict[str, Any]:
    return {
        "doc_id": rank.doc.doc_id,
        "source_type": rank.doc.source_type,
        "text": rank.doc.text,
        "metadata": rank.doc.metadata,
        "score": round(rank.score, 4),
        "reasons": rank.reasons,
    }


def build_audit_app(config: DemoConfig):
    docs: list[SourceDocument] = load_all_sources(config.data_dir)
    retriever = GraphRAGRetriever(docs, graph_hop_boost=config.graph_hop_boost)

    llm = ChatOllama(model=config.model_name, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial audit copilot. Use only provided evidence. "
                "If evidence is weak, say so. Always cite sources in format [source_type:doc_id].",
            ),
            (
                "human",
                "Question: {query}\n\nEvidence:\n{evidence}\n\n"
                "Respond with:\n"
                "1) concise answer\n2) key risk signals\n3) next audit actions\n"
                "Include citations.",
            ),
        ]
    )

    def router_node(state: AuditState) -> AuditState:
        selected = route_sources(state["query"])
        return {"selected_sources": selected}

    def retrieve_node(state: AuditState) -> AuditState:
        selected = state.get("selected_sources", [])
        ranked = retriever.retrieve(
            query=state["query"],
            source_types=selected,
            top_k=max(3, config.top_k_per_source * max(len(selected), 1)),
        )
        return {"retrieved": [_to_citation(r) for r in ranked]}

    def synthesize_node(state: AuditState) -> AuditState:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {"answer": "No evidence found for the query in current sources."}

        evidence_lines = []
        for item in retrieved:
            evidence_lines.append(
                f"- [{item['source_type']}:{item['doc_id']}] score={item['score']} "
                f"reasons={','.join(item.get('reasons', [])) or 'lexical'}\n  {item['text']}"
            )
        evidence_blob = "\n".join(evidence_lines)

        chain = prompt | llm
        out = chain.invoke({"query": state["query"], "evidence": evidence_blob})
        return {"answer": out.content}

    graph = StateGraph(AuditState)
    graph.add_node("route", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
