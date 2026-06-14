from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    source_type: str
    text: str
    metadata: dict[str, Any]


def load_transactions(data_dir: str) -> list[SourceDocument]:
    path = Path(data_dir) / "transactions.csv"
    df = pd.read_csv(path)
    docs: list[SourceDocument] = []

    for row in df.to_dict(orient="records"):
        txt = (
            f"Transaction {row['txn_id']} on {row['date']} for vendor {row['vendor']} "
            f"account {row['account']} employee {row['employee']} amount {row['amount']} {row['currency']} "
            f"country {row['country']} risk_tag {row['risk_tag']} description: {row['description']}"
        )
        docs.append(
            SourceDocument(
                doc_id=row["txn_id"],
                source_type="transactions",
                text=txt,
                metadata=row,
            )
        )
    return docs


def load_audit_logs(data_dir: str) -> list[SourceDocument]:
    path = Path(data_dir) / "audit_logs.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: list[SourceDocument] = []

    for row in raw:
        txt = (
            f"Audit log {row['log_id']} at {row['timestamp']} severity {row['severity']} "
            f"control {row['control_id']} linked_txn {row['linked_txn']} "
            f"message: {row['message']} note: {row['analyst_note']}"
        )
        docs.append(
            SourceDocument(
                doc_id=row["log_id"],
                source_type="audit_logs",
                text=txt,
                metadata=row,
            )
        )
    return docs


def load_financial_statements(data_dir: str) -> list[SourceDocument]:
    path = Path(data_dir) / "financial_statements.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: list[SourceDocument] = []

    for row in raw:
        txt = (
            f"Financial statement {row['statement_id']} period {row['period']} revenue {row['revenue']} "
            f"expense {row['expense']} operating_margin {row['operating_margin']} cash_flow_ops {row['cash_flow_ops']} "
            f"note: {row['note']}"
        )
        docs.append(
            SourceDocument(
                doc_id=row["statement_id"],
                source_type="financial_statements",
                text=txt,
                metadata=row,
            )
        )
    return docs


def load_all_sources(data_dir: str) -> list[SourceDocument]:
    return (
        load_transactions(data_dir)
        + load_audit_logs(data_dir)
        + load_financial_statements(data_dir)
    )
