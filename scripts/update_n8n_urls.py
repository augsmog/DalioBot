#!/usr/bin/env python3
"""
Update existing DalioBot n8n workflows to point to the Render-deployed API URL.
Run this after deploying to Render.

Usage: python scripts/update_n8n_urls.py https://daliobot-api.onrender.com
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

N8N_URL = os.getenv("N8N_API_URL", "").rstrip("/")
N8N_API_KEY = os.getenv("N8N_API_KEY", "")

HEADERS = {
    "X-N8N-API-KEY": N8N_API_KEY,
    "Content-Type": "application/json",
}

DALIOBOT_WORKFLOW_PREFIX = "DalioBot"


def get_daliobot_workflows() -> list:
    """Find all DalioBot workflows."""
    resp = requests.get(f"{N8N_URL}/api/v1/workflows", headers=HEADERS)
    if resp.status_code != 200:
        print(f"Failed to list workflows: {resp.status_code}")
        return []

    workflows = resp.json().get("data", [])
    return [w for w in workflows if w.get("name", "").startswith(DALIOBOT_WORKFLOW_PREFIX)]


def update_workflow_urls(workflow_id: str, workflow_name: str, new_api_url: str) -> bool:
    """Update all HTTP request nodes in a workflow to use the new API URL."""
    # Get full workflow
    resp = requests.get(f"{N8N_URL}/api/v1/workflows/{workflow_id}", headers=HEADERS)
    if resp.status_code != 200:
        print(f"  Failed to get workflow {workflow_id}: {resp.status_code}")
        return False

    workflow = resp.json()
    nodes = workflow.get("nodes", [])
    updated = False

    for node in nodes:
        if node.get("type") == "n8n-nodes-base.httpRequest":
            params = node.get("parameters", {})
            url = params.get("url", "")
            if "localhost" in url or "daliobot" in url.lower():
                # Replace the base URL
                old_url = url
                # Extract the path (e.g., /webhook/morning)
                path = url.split("/webhook/")[-1] if "/webhook/" in url else ""
                new_url = f"{new_api_url.rstrip('/')}/webhook/{path}"
                params["url"] = new_url
                updated = True
                print(f"  {node['name']}: {old_url} → {new_url}")

    if updated:
        # Update the workflow
        update_payload = {
            "name": workflow.get("name", workflow_name),
            "nodes": nodes,
            "connections": workflow.get("connections", {}),
            "settings": workflow.get("settings", {}),
        }
        resp = requests.put(
            f"{N8N_URL}/api/v1/workflows/{workflow_id}",
            headers=HEADERS,
            json=update_payload,
        )
        if resp.status_code == 200:
            print(f"  Updated: {workflow_name}")
            return True
        else:
            print(f"  FAILED to update: {resp.status_code} {resp.text[:200]}")
            return False
    else:
        print(f"  No HTTP nodes to update in {workflow_name}")
        return False


def activate_workflow(workflow_id: str, workflow_name: str) -> bool:
    """Activate a workflow."""
    resp = requests.post(
        f"{N8N_URL}/api/v1/workflows/{workflow_id}/activate",
        headers=HEADERS,
    )
    if resp.status_code == 200:
        print(f"  Activated: {workflow_name}")
        return True
    else:
        print(f"  Failed to activate {workflow_name}: {resp.status_code} {resp.text[:200]}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_n8n_urls.py <RENDER_URL>")
        print("Example: python scripts/update_n8n_urls.py https://daliobot-api.onrender.com")
        sys.exit(1)

    new_api_url = sys.argv[1].rstrip("/")
    print(f"Updating DalioBot n8n workflows to: {new_api_url}")
    print("=" * 50)

    # Verify the new API is reachable
    try:
        resp = requests.get(f"{new_api_url}/health", timeout=10)
        if resp.status_code == 200:
            print(f"API health check: OK")
        else:
            print(f"WARNING: API health check returned {resp.status_code}")
    except Exception as e:
        print(f"WARNING: Could not reach {new_api_url}: {e}")
        print("Continuing anyway (Render may still be starting up)...")

    workflows = get_daliobot_workflows()
    if not workflows:
        print("No DalioBot workflows found in n8n.")
        return

    print(f"\nFound {len(workflows)} DalioBot workflows:\n")

    for w in workflows:
        wid = w["id"]
        wname = w["name"]
        print(f"\n{wname} (ID: {wid})")
        update_workflow_urls(wid, wname, new_api_url)

    # Ask to activate
    print("\n" + "=" * 50)
    activate = input("Activate all DalioBot workflows? (y/n): ").strip().lower()
    if activate == "y":
        for w in workflows:
            activate_workflow(w["id"], w["name"])
        print("\nAll workflows activated! The self-improvement loop is now running 24/7.")
    else:
        print("\nWorkflows updated but NOT activated. Activate manually in n8n dashboard.")


if __name__ == "__main__":
    main()
