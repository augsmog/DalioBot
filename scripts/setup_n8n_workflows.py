#!/usr/bin/env python3
"""
Create DalioBot automation workflows in n8n.

Workflows:
1. Morning Routine — runs at 9:55 AM ET every weekday
2. ML Self-Improvement — runs every 6 hours
3. Weekend Backtest — runs Saturday morning
4. Emergency Monitor — checks drawdown every 15 min during market hours
"""

import json
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

N8N_URL = os.getenv("N8N_API_URL", "").rstrip("/")
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
WEBHOOK_SECRET = os.getenv("N8N_WEBHOOK_SECRET", "")

# DalioBot API base URL — update this when you deploy or use a tunnel
DALIOBOT_API = os.getenv("DALIOBOT_API_URL", "http://localhost:8000")

HEADERS = {
    "X-N8N-API-KEY": N8N_API_KEY,
    "Content-Type": "application/json",
}


def create_workflow(name: str, nodes: list, connections: dict, settings: dict = None) -> dict:
    """Create a workflow via n8n API."""
    payload = {
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "settings": settings or {"executionOrder": "v1"},
    }

    resp = requests.post(f"{N8N_URL}/api/v1/workflows", headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        data = resp.json()
        print(f"  Created: {name} (ID: {data.get('id', 'unknown')})")
        return data
    else:
        print(f"  FAILED: {name} — {resp.status_code}: {resp.text[:200]}")
        return {}


def build_morning_routine_workflow():
    """Morning Routine — 9:55 AM ET weekdays."""
    nodes = [
        {
            "parameters": {"rule": {
                "interval": [{"field": "cronExpression", "expression": "55 9 * * 1-5"}]
            }},
            "id": "morning-trigger",
            "name": "Weekday 9:55 AM",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [0, 0],
        },
        {
            "parameters": {
                "url": f"{DALIOBOT_API}/webhook/morning",
                "sendBody": True,
                "method": "POST",
                "options": {"timeout": 120000},
            },
            "id": "call-morning",
            "name": "Run Morning Routine",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [220, 0],
        },
        {
            "parameters": {
                "conditions": {
                    "options": {"version": 2},
                    "combinator": "and",
                    "conditions": [{
                        "leftValue": "={{ $json.action }}",
                        "rightValue": "SIGNALS_GENERATED",
                        "operator": {"type": "string", "operation": "equals"},
                    }],
                },
            },
            "id": "check-signals",
            "name": "Has Signals?",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2,
            "position": [440, 0],
        },
        {
            "parameters": {
                "jsCode": """
// Format trade signal for notification
const data = $input.all()[0].json;
const signals = data.signals || [];
const regime = data.regime || {};

let message = `🤖 DALIOBOT MORNING REPORT\\n`;
message += `━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;
message += `Capital: $${data.capital?.toLocaleString() || 'N/A'}\\n`;
message += `Regime: ${regime.regime?.toUpperCase() || 'N/A'} (${(regime.confidence * 100)?.toFixed(0) || '?'}%)\\n`;
message += `\\n📊 TRADE SIGNALS:\\n`;

for (const sig of signals) {
    message += `\\n▸ ${sig.action}: ${sig.ticker}\\n`;
    message += `  Strike: $${sig.strike} | Expiry: ${sig.expiry}\\n`;
    message += `  Premium: $${sig.premium?.toFixed(2)} | Max Loss: $${sig.max_loss?.toFixed(2)}\\n`;
    message += `  Confidence: ${(sig.confidence * 100)?.toFixed(0)}%\\n`;
}

return [{ json: { message, ...data } }];
""",
            },
            "id": "format-signal",
            "name": "Format Signal Alert",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [660, -100],
        },
        {
            "parameters": {
                "jsCode": """
const data = $input.all()[0].json;
const regime = data.regime || {};
const risk = data.risk || {};

let message = `🤖 DALIOBOT: No trades today\\n`;
message += `Regime: ${regime.regime?.toUpperCase() || 'N/A'}\\n`;
message += `Risk: ${risk.reason || 'All clear'}\\n`;
message += `Action: ${data.action || 'WAITING'}`;

return [{ json: { message, ...data } }];
""",
            },
            "id": "format-no-signal",
            "name": "Format No-Signal",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [660, 100],
        },
    ]

    connections = {
        "Weekday 9:55 AM": {"main": [[{"node": "Run Morning Routine", "type": "main", "index": 0}]]},
        "Run Morning Routine": {"main": [[{"node": "Has Signals?", "type": "main", "index": 0}]]},
        "Has Signals?": {
            "main": [
                [{"node": "Format Signal Alert", "type": "main", "index": 0}],
                [{"node": "Format No-Signal", "type": "main", "index": 0}],
            ]
        },
    }

    return create_workflow("DalioBot — Morning Routine (9:55 AM)", nodes, connections)


def build_ml_training_workflow():
    """ML Self-Improvement — every 6 hours."""
    nodes = [
        {
            "parameters": {"rule": {
                "interval": [{"field": "hours", "hoursInterval": 6}]
            }},
            "id": "train-trigger",
            "name": "Every 6 Hours",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [0, 0],
        },
        {
            "parameters": {
                "url": f"{DALIOBOT_API}/webhook/train",
                "method": "POST",
                "options": {"timeout": 300000},
            },
            "id": "call-train",
            "name": "Run ML Training",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [220, 0],
        },
        {
            "parameters": {
                "jsCode": """
const data = $input.all()[0].json;
const models = data.models || [];

let message = `🧠 ML TRAINING COMPLETE\\n`;
message += `━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;

for (const m of models) {
    const pred = m.todays_prediction || {};
    message += `\\n${m.ticker}:`;
    message += ` Acc=${(m.base_accuracy * 100)?.toFixed(1)}%`;
    message += ` | Improvements: ${m.improvements_found}/10`;
    message += ` | Signal: ${pred.signal || 'N/A'} (${(pred.confidence * 100)?.toFixed(0) || '?'}%)\\n`;
}

return [{ json: { message, ...data } }];
""",
            },
            "id": "format-training",
            "name": "Format Training Results",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [440, 0],
        },
    ]

    connections = {
        "Every 6 Hours": {"main": [[{"node": "Run ML Training", "type": "main", "index": 0}]]},
        "Run ML Training": {"main": [[{"node": "Format Training Results", "type": "main", "index": 0}]]},
    }

    return create_workflow("DalioBot — ML Self-Improvement (6h)", nodes, connections)


def build_weekend_backtest_workflow():
    """Weekend Backtest — Saturday 8 AM."""
    nodes = [
        {
            "parameters": {"rule": {
                "interval": [{"field": "cronExpression", "expression": "0 8 * * 6"}]
            }},
            "id": "backtest-trigger",
            "name": "Saturday 8 AM",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [0, 0],
        },
        {
            "parameters": {
                "url": f"{DALIOBOT_API}/webhook/backtest",
                "method": "POST",
                "options": {"timeout": 300000},
            },
            "id": "call-backtest",
            "name": "Run Backtest",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [220, 0],
        },
        {
            "parameters": {
                "jsCode": """
const data = $input.all()[0].json;
const results = data.results || [];

let message = `📊 WEEKLY BACKTEST REPORT\\n`;
message += `━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;

for (const r of results) {
    message += `\\n${r.ticker}:`;
    message += ` Return=${r.total_return_pct?.toFixed(1)}%`;
    message += ` | Win=${r.win_rate?.toFixed(0)}%`;
    message += ` | Sharpe=${r.sharpe_ratio?.toFixed(2)}`;
    message += ` | MaxDD=${r.max_drawdown_pct?.toFixed(1)}%\\n`;
}

return [{ json: { message, ...data } }];
""",
            },
            "id": "format-backtest",
            "name": "Format Backtest Report",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [440, 0],
        },
    ]

    connections = {
        "Saturday 8 AM": {"main": [[{"node": "Run Backtest", "type": "main", "index": 0}]]},
        "Run Backtest": {"main": [[{"node": "Format Backtest Report", "type": "main", "index": 0}]]},
    }

    return create_workflow("DalioBot — Weekend Backtest", nodes, connections)


def build_risk_monitor_workflow():
    """Risk Monitor — every 15 min during market hours."""
    nodes = [
        {
            "parameters": {"rule": {
                "interval": [{"field": "cronExpression", "expression": "*/15 9-16 * * 1-5"}]
            }},
            "id": "risk-trigger",
            "name": "Every 15 Min (Market Hours)",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [0, 0],
        },
        {
            "parameters": {
                "url": f"{DALIOBOT_API}/webhook/dashboard",
                "method": "POST",
                "options": {"timeout": 60000},
            },
            "id": "call-dashboard",
            "name": "Check Dashboard",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [220, 0],
        },
        {
            "parameters": {
                "conditions": {
                    "options": {"version": 2},
                    "combinator": "or",
                    "conditions": [
                        {
                            "leftValue": "={{ $json.risk.can_trade }}",
                            "rightValue": False,
                            "operator": {"type": "boolean", "operation": "equals"},
                        },
                        {
                            "leftValue": "={{ $json.regime.regime }}",
                            "rightValue": "crisis",
                            "operator": {"type": "string", "operation": "equals"},
                        },
                    ],
                },
            },
            "id": "check-alert",
            "name": "Alert Needed?",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2,
            "position": [440, 0],
        },
        {
            "parameters": {
                "jsCode": """
const data = $input.all()[0].json;
const risk = data.risk || {};
const regime = data.regime || {};

let message = `🚨 DALIOBOT RISK ALERT\\n`;
message += `━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;
message += `Status: ${risk.can_trade ? 'CAN TRADE' : 'HALTED'}\\n`;
message += `Reason: ${risk.reason}\\n`;
message += `Drawdown: ${risk.drawdown_pct}%\\n`;
message += `Regime: ${regime.regime?.toUpperCase()} — ${regime.rationale}\\n`;
message += `\\nCapital: $${data.capital?.toLocaleString()}`;

return [{ json: { message, alert: true, ...data } }];
""",
            },
            "id": "format-alert",
            "name": "Format Risk Alert",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [660, -100],
        },
        {
            "parameters": {},
            "id": "no-alert",
            "name": "All Clear",
            "type": "n8n-nodes-base.noOp",
            "typeVersion": 1,
            "position": [660, 100],
        },
    ]

    connections = {
        "Every 15 Min (Market Hours)": {"main": [[{"node": "Check Dashboard", "type": "main", "index": 0}]]},
        "Check Dashboard": {"main": [[{"node": "Alert Needed?", "type": "main", "index": 0}]]},
        "Alert Needed?": {
            "main": [
                [{"node": "Format Risk Alert", "type": "main", "index": 0}],
                [{"node": "All Clear", "type": "main", "index": 0}],
            ]
        },
    }

    return create_workflow("DalioBot — Risk Monitor (15m)", nodes, connections)


def main():
    if not N8N_URL or not N8N_API_KEY:
        print("ERROR: N8N_API_URL and N8N_API_KEY must be set in .env")
        return

    print(f"Creating DalioBot workflows on n8n: {N8N_URL}")
    print(f"DalioBot API target: {DALIOBOT_API}")
    print("=" * 50)

    workflows = []

    print("\n1. Morning Routine (9:55 AM ET, weekdays)")
    w = build_morning_routine_workflow()
    if w:
        workflows.append(w)

    print("\n2. ML Self-Improvement (every 6 hours)")
    w = build_ml_training_workflow()
    if w:
        workflows.append(w)

    print("\n3. Weekend Backtest (Saturday 8 AM)")
    w = build_weekend_backtest_workflow()
    if w:
        workflows.append(w)

    print("\n4. Risk Monitor (every 15 min, market hours)")
    w = build_risk_monitor_workflow()
    if w:
        workflows.append(w)

    print("\n" + "=" * 50)
    print(f"Created {len(workflows)} workflows.")
    print("\nIMPORTANT: Workflows are created INACTIVE.")
    print("Review them in n8n dashboard, then activate when ready.")
    print(f"\nDashboard: {N8N_URL}")

    if workflows:
        print("\nWorkflow IDs:")
        for w in workflows:
            print(f"  - {w.get('name', 'unknown')}: {w.get('id', 'unknown')}")


if __name__ == "__main__":
    main()
