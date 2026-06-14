import argparse
import csv
import json
import os
import random
from datetime import datetime, timedelta


def _rand_date(days_back: int = 365) -> str:
    d = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")


def generate_transactions(path: str, n: int) -> None:
    vendors = [
        "Apex Supplies",
        "Northwind Metals",
        "BlueWave Consulting",
        "Orion Freight",
        "Nimbus IT",
        "Rapid Retail",
        "Helios Energy",
        "Vertex Labs",
        "Summit Services",
        "Keystone Partners",
    ]
    accounts = ["AP", "AR", "CASH", "PAYROLL", "OPS", "TRAVEL", "CAPEX"]
    employees = [f"EMP-{100+i}" for i in range(45)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "txn_id",
                "date",
                "vendor",
                "account",
                "employee",
                "amount",
                "currency",
                "country",
                "description",
                "risk_tag",
            ],
        )
        w.writeheader()
        for i in range(n):
            vendor = random.choice(vendors)
            account = random.choice(accounts)
            employee = random.choice(employees)
            amount = round(abs(random.gauss(2200, 1800)) + 40, 2)
            risk_tag = "normal"

            if random.random() < 0.06:
                amount = round(amount * random.uniform(4.0, 18.0), 2)
                risk_tag = random.choice(["duplicate-risk", "split-risk", "round-number"])

            desc = f"Invoice settlement for {vendor} in {account}"
            if risk_tag != "normal":
                desc = f"Potential anomaly: {risk_tag} linked to {vendor}"

            w.writerow(
                {
                    "txn_id": f"TXN-{i+1:05d}",
                    "date": _rand_date(),
                    "vendor": vendor,
                    "account": account,
                    "employee": employee,
                    "amount": amount,
                    "currency": "USD",
                    "country": random.choice(["US", "UK", "DE", "IN", "SG", "AE"]),
                    "description": desc,
                    "risk_tag": risk_tag,
                }
            )


def generate_audit_logs(path: str, n: int) -> None:
    severities = ["low", "medium", "high", "critical"]
    controls = ["C-101", "C-145", "C-210", "C-301", "C-333", "C-404"]

    logs = []
    for i in range(n):
        sev = random.choices(severities, weights=[0.5, 0.25, 0.18, 0.07], k=1)[0]
        linked_txn = f"TXN-{random.randint(1, 1000):05d}"
        control = random.choice(controls)
        logs.append(
            {
                "log_id": f"LOG-{i+1:04d}",
                "timestamp": f"{_rand_date()}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z",
                "severity": sev,
                "control_id": control,
                "linked_txn": linked_txn,
                "message": f"{sev.upper()} exception detected for {linked_txn}; control {control} review required.",
                "analyst_note": random.choice(
                    [
                        "Needs manual validation.",
                        "Pattern resembles prior vendor fraud case.",
                        "Possible false positive due to period close.",
                        "Escalate to internal controls team.",
                    ]
                ),
            }
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def generate_financial_statements(path: str, months: int = 12) -> None:
    items = []
    base_revenue = 4200000
    base_expense = 2900000

    for i in range(months):
        month_date = datetime.utcnow() - timedelta(days=30 * i)
        period = month_date.strftime("%Y-%m")
        revenue = base_revenue + random.randint(-350000, 420000)
        expense = base_expense + random.randint(-250000, 370000)
        op_margin = round((revenue - expense) / revenue, 4)

        items.append(
            {
                "statement_id": f"FS-{period}",
                "period": period,
                "revenue": revenue,
                "expense": expense,
                "operating_margin": op_margin,
                "cash_flow_ops": random.randint(300000, 900000),
                "note": random.choice(
                    [
                        "Higher freight costs impacted margin.",
                        "Seasonal demand improved top line.",
                        "One-time vendor settlement recognized.",
                        "Payroll optimization reduced overhead.",
                    ]
                ),
            }
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FinAudit data")
    parser.add_argument("--out", default="data/synthetic", help="Output folder")
    parser.add_argument("--transactions", type=int, default=1000)
    parser.add_argument("--audit-logs", type=int, default=180)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    generate_transactions(os.path.join(args.out, "transactions.csv"), args.transactions)
    generate_audit_logs(os.path.join(args.out, "audit_logs.json"), args.audit_logs)
    generate_financial_statements(os.path.join(args.out, "financial_statements.json"), args.months)

    print(f"Synthetic data generated in: {args.out}")


if __name__ == "__main__":
    main()
