"""
test_bot.py - End-to-End Automated Verification Test Suite
==========================================================
Runs the complete testing matrix replicating judge_simulator.py:
- Warmup: healthz, metadata, context ingestion, version conflict (409)
- Auto-reply hell: canned message detection and graceful exit
- Intent transition: commitment detection, zero qualifying phrases
- Hostile merchant: opt-out detection, conversation termination
- Curveball / out-of-scope redirection
- Tick proactive composition: 4-context assembly, 0 URL violations, valid CTA
- Submission format verification: 30 test pairs, complete schema
"""

from __future__ import annotations
import json
from pathlib import Path
from fastapi.testclient import TestClient
import bot
from context_store import global_context_store

client = TestClient(bot.app)

def test_all():
    print("=" * 60)
    print("RUNNING COMPLETE BOT VERIFICATION SUITE")
    print("=" * 60)

    # 1. Healthz & Metadata
    res = client.get("/v1/healthz")
    assert res.status_code == 200, f"healthz failed: {res.text}"
    data = res.json()
    assert data["status"] == "ok"
    print("PASS: /v1/healthz")

    res = client.get("/v1/metadata")
    assert res.status_code == 200, f"metadata failed: {res.text}"
    meta = res.json()
    assert "team_name" in meta and "model" in meta
    print("PASS: /v1/metadata")

    # 2. Context Ingestion & Idempotency
    global_context_store.clear()
    cat_payload = {
        "slug": "dentists",
        "voice": {"tone": "peer_clinical"},
        "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]
    }
    # Initial push -> 200
    res = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1, "payload": cat_payload
    })
    assert res.status_code == 200 and res.json()["accepted"] is True
    print("PASS: /v1/context initial push accepted (200)")

    # Idempotent re-push of same version -> 200 (no-op)
    res = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1, "payload": cat_payload
    })
    assert res.status_code == 200 and res.json()["accepted"] is True
    print("PASS: /v1/context idempotent same-version re-push accepted (200)")

    # Higher version push -> 200
    cat_payload["version_2_note"] = "updated"
    res = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 2, "payload": cat_payload
    })
    assert res.status_code == 200 and res.json()["accepted"] is True
    print("PASS: /v1/context higher version replaces atomically (200)")

    # Stale lower version push (v1 when v2 exists) -> 409
    res = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1, "payload": cat_payload
    })
    assert res.status_code == 409 and res.json()["accepted"] is False
    print("PASS: /v1/context stale version rejected (409)")

    # 3. Ingest Seeds & Expanded Dataset for Full Test
    root = Path(__file__).parent
    dataset_dir = root / "dataset"
    expanded_dir = dataset_dir / "expanded"

    # Ingest categories
    for cf in (dataset_dir / "categories").glob("*.json"):
        with open(cf) as fp:
            cdata = json.load(fp)
            client.post("/v1/context", json={
                "scope": "category", "context_id": cdata["slug"], "version": 1, "payload": cdata
            })

    # Ingest sample merchants, customers, triggers
    merchants_loaded = 0
    for mf in list((expanded_dir / "merchants").glob("*.json"))[:10]:
        with open(mf) as fp:
            mdata = json.load(fp)
            client.post("/v1/context", json={
                "scope": "merchant", "context_id": mdata["merchant_id"], "version": 1, "payload": mdata
            })
            merchants_loaded += 1

    sample_triggers = []
    for tf in list((expanded_dir / "triggers").glob("*.json"))[:5]:
        with open(tf) as fp:
            tdata = json.load(fp)
            client.post("/v1/context", json={
                "scope": "trigger", "context_id": tdata["id"], "version": 1, "payload": tdata
            })
            sample_triggers.append(tdata["id"])

    res = client.get("/v1/healthz")
    counts = res.json()["contexts_loaded"]
    assert counts["category"] >= 5 and counts["merchant"] >= 10
    print(f"PASS: Context counts verified: {counts}")

    # 4. Proactive Tick Composition
    res = client.post("/v1/tick", json={
        "now": "2026-09-27T10:00:00Z",
        "available_triggers": sample_triggers
    })
    assert res.status_code == 200
    actions = res.json()["actions"]
    print(f"PASS: /v1/tick returned {len(actions)} actions")
    for act in actions:
        assert "body" in act and len(act["body"]) > 20
        assert "cta" in act
        assert "http://" not in act["body"] and "https://" not in act["body"], "No URLs allowed in message body"
        assert act["send_as"] in ("vera", "merchant_on_behalf")

    # 5. Multi-Turn: Auto-Reply Detection
    auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."
    res1 = client.post("/v1/reply", json={
        "conversation_id": "conv_auto_test",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": auto_msg,
        "turn_number": 2
    })
    assert res1.status_code == 200
    action1 = res1.json()["action"]
    assert action1 in ("wait", "end"), f"Expected wait or end on auto-reply, got {action1}"
    print(f"PASS: Auto-reply Turn 1 returned action='{action1}'")

    res2 = client.post("/v1/reply", json={
        "conversation_id": "conv_auto_test",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": auto_msg,
        "turn_number": 3
    })
    assert res2.status_code == 200
    action2 = res2.json()["action"]
    assert action2 == "end", f"Expected end on repeated auto-reply, got {action2}"
    print(f"PASS: Repeated Auto-reply Turn 2 returned action='{action2}'")

    # 6. Multi-Turn: Intent Transition
    commitment = "Ok lets do it. Whats next?"
    res = client.post("/v1/reply", json={
        "conversation_id": "conv_intent_test",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": commitment,
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "send"
    body_lower = data["body"].lower()

    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]

    assert any(w in body_lower for w in actioning), f"Expected action words in {body_lower}"
    assert not any(w in body_lower for w in qualifying), f"Qualifying words found in {body_lower}"
    print("PASS: Intent transition switched immediately to action mode with 0 qualifying phrases")

    # 7. Multi-Turn: Hostility & Opt-Out
    hostile = "Stop messaging me. This is useless spam."
    res = client.post("/v1/reply", json={
        "conversation_id": "conv_hostile_test",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": hostile,
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "end"
    print("PASS: Hostile message properly handled with action='end'")

    # 8. Multi-Turn: Out-of-Scope Redirection
    curveball = "Can you also help me file my GST returns?"
    res = client.post("/v1/reply", json={
        "conversation_id": "conv_curveball_test",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": curveball,
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "send"
    assert "gst" in data["body"].lower() or "accountant" in data["body"].lower()
    print("PASS: Out-of-scope question politely handled with topic redirection")

    # 9. Verify submission.jsonl
    sub_file = root / "submission.jsonl"
    assert sub_file.exists(), "submission.jsonl not found"
    lines = sub_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 30, f"Expected 30 lines in submission.jsonl, got {len(lines)}"

    required_keys = {"test_id", "category", "merchant_id", "trigger_type", "customer_id", "composed_message", "cta", "lever_used", "facts_used", "language"}
    for idx, line in enumerate(lines, 1):
        obj = json.loads(line)
        for k in required_keys:
            assert k in obj, f"Key '{k}' missing from line {idx}"
    print(f"PASS: submission.jsonl contains 30 valid entries with all required keys")

    print("=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
