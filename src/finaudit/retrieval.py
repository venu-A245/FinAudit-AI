from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from .data_loader import SourceDocument


TOKEN_RE = re.compile(r"[a-zA-Z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RankedDocument:
    doc: SourceDocument
    score: float
    reasons: list[str]


class GraphRAGRetriever:
    def __init__(self, docs: list[SourceDocument], graph_hop_boost: float = 0.12):
        self.docs = docs
        self.graph_hop_boost = graph_hop_boost
        self.by_source: dict[str, list[SourceDocument]] = defaultdict(list)
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_tf: dict[str, Counter[str]] = {}
        self.idf: dict[str, float] = {}
        self.graph = nx.Graph()

        for d in docs:
            self.by_source[d.source_type].append(d)
            toks = _tokenize(d.text)
            self.doc_tokens[d.doc_id] = toks
            self.doc_tf[d.doc_id] = Counter(toks)

        self._build_idf()
        self._build_graph()

    def _build_idf(self) -> None:
        df = Counter()
        n_docs = len(self.docs)
        for d in self.docs:
            for tok in set(self.doc_tokens[d.doc_id]):
                df[tok] += 1

        for tok, freq in df.items():
            self.idf[tok] = math.log((1 + n_docs) / (1 + freq)) + 1.0

    def _build_graph(self) -> None:
        for d in self.docs:
            self.graph.add_node(d.doc_id, kind="doc", source=d.source_type)
            entities = self._extract_entities(d)
            for ent in entities:
                ent_node = f"ENT::{ent}"
                self.graph.add_node(ent_node, kind="entity")
                self.graph.add_edge(d.doc_id, ent_node)

    def _extract_entities(self, d: SourceDocument) -> set[str]:
        md = d.metadata
        ents = set()
        for k in ["txn_id", "vendor", "account", "employee", "linked_txn", "control_id", "period"]:
            v = md.get(k)
            if v is not None:
                ents.add(str(v).lower())
        return ents

    def _score(self, q_tokens: list[str], doc: SourceDocument) -> float:
        tf = self.doc_tf[doc.doc_id]
        score = 0.0
        for t in q_tokens:
            if t in tf:
                score += (1 + math.log(tf[t])) * self.idf.get(t, 1.0)
        return score

    def _graph_boost(self, query: str, doc_id: str) -> tuple[float, list[str]]:
        q_tokens = set(_tokenize(query))
        reasons: list[str] = []
        boost = 0.0

        for n in self.graph.neighbors(doc_id):
            if n.startswith("ENT::"):
                ent = n.replace("ENT::", "")
                if ent in q_tokens:
                    boost += self.graph_hop_boost
                    reasons.append(f"entity-match:{ent}")
                else:
                    # One-hop traversal: if neighbor docs share this entity, add small relation score.
                    related_docs = [x for x in self.graph.neighbors(n) if x != doc_id]
                    if related_docs:
                        boost += self.graph_hop_boost / 3
        return boost, reasons

    def retrieve(
        self,
        query: str,
        source_types: Iterable[str] | None = None,
        top_k: int = 5,
    ) -> list[RankedDocument]:
        q_tokens = _tokenize(query)
        scoped_docs = self.docs
        if source_types:
            source_set = set(source_types)
            scoped_docs = [d for d in self.docs if d.source_type in source_set]

        ranked: list[RankedDocument] = []
        for d in scoped_docs:
            lexical = self._score(q_tokens, d)
            boost, reasons = self._graph_boost(query, d.doc_id)
            total = lexical + boost
            if total > 0:
                ranked.append(RankedDocument(doc=d, score=total, reasons=reasons))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
