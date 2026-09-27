"""
context_store.py - High Performance In-Memory Context & State Store
===================================================================
Manages Category, Merchant, Customer, and Trigger contexts with:
- Idempotency & version conflict detection (atomic replaces)
- Conversation history tracking per merchant & conversation ID
- Trigger suppression & cooldown cache
- Fast O(1) lookups for the composition engine
"""

from __future__ import annotations
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class ContextStore:
    def __init__(self):
        self._lock = threading.RLock()
        # Key: (scope, context_id) -> {"version": int, "payload": dict, "stored_at": str}
        self._contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Conversation state: conversation_id -> list of turn dicts
        # Each turn: {"from": str, "message": str, "timestamp": str, "action": Optional[dict]}
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}

        # Merchant to conversation_ids mapping
        self._merchant_conversations: Dict[str, List[str]] = {}

        # Suppression keys: suppression_key -> datetime of last send
        self._suppressed_keys: Dict[str, str] = {}

        # Opted-out / hostile merchants: merchant_id -> timestamp
        self._opted_out_merchants: Dict[str, str] = {}

        # Track sent messages per conversation to avoid repetition penalty
        self._sent_message_hashes: Dict[str, set] = {}

    def push_context(self, scope: str, context_id: str, version: int, payload: Dict[str, Any], delivered_at: Optional[str] = None) -> Tuple[bool, Dict[str, Any], int]:
        """
        Idempotent context upsert.
        Returns: (accepted, response_body, http_status_code)
        """
        valid_scopes = {"category", "merchant", "customer", "trigger"}
        if scope not in valid_scopes:
            return False, {"accepted": False, "reason": "invalid_scope", "details": f"Scope must be one of {valid_scopes}"}, 400

        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        key = (scope, context_id)

        with self._lock:
            existing = self._contexts.get(key)
            if existing is not None:
                cur_version = existing["version"]
                if version < cur_version:
                    # Version conflict: you already have a higher version
                    return False, {
                        "accepted": False,
                        "reason": "stale_version",
                        "current_version": cur_version
                    }, 409
                elif version == cur_version:
                    if scope == "merchant":
                        self._opted_out_merchants.pop(context_id, None)
                    if scope == "trigger":
                        supp_key = payload.get("suppression_key")
                        if supp_key:
                            self._suppressed_keys.pop(supp_key, None)
                        merchant_id = payload.get("merchant_id", "")
                        conv_id = f"conv_{merchant_id}_{context_id}"
                        self._sent_message_hashes.pop(conv_id, None)
                    return True, {
                        "accepted": True,
                        "ack_id": f"ack_{context_id}_v{version}",
                        "stored_at": existing.get("stored_at", now_iso)
                    }, 200

            # Higher version replaces atomically, or new context inserted
            ack_id = f"ack_{context_id}_v{version}"
            self._contexts[key] = {
                "version": version,
                "payload": payload,
                "stored_at": now_iso,
                "delivered_at": delivered_at or now_iso
            }

            if scope == "merchant":
                self._opted_out_merchants.pop(context_id, None)

            if scope == "trigger":
                supp_key = payload.get("suppression_key")
                if supp_key:
                    self._suppressed_keys.pop(supp_key, None)
                merchant_id = payload.get("merchant_id", "")
                conv_id = f"conv_{merchant_id}_{context_id}"
                self._sent_message_hashes.pop(conv_id, None)

            return True, {
                "accepted": True,
                "ack_id": ack_id,
                "stored_at": now_iso
            }, 200

    def get_context(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._contexts.get((scope, context_id))
            return item["payload"] if item else None

    def get_category(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.get_context("category", slug)

    def get_merchant(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("merchant", merchant_id)

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("customer", customer_id)

    def get_trigger(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("trigger", trigger_id)

    def get_counts(self) -> Dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _), _ in self._contexts.items():
                if scope in counts:
                    counts[scope] += 1
        return counts

    def is_suppressed(self, suppression_key: Optional[str]) -> bool:
        if not suppression_key:
            return False
        with self._lock:
            return suppression_key in self._suppressed_keys

    def record_suppression(self, suppression_key: Optional[str]):
        if suppression_key:
            with self._lock:
                self._suppressed_keys[suppression_key] = datetime.now(timezone.utc).isoformat()

    def is_merchant_opted_out(self, merchant_id: str) -> bool:
        with self._lock:
            return merchant_id in self._opted_out_merchants

    def opt_out_merchant(self, merchant_id: str):
        with self._lock:
            self._opted_out_merchants[merchant_id] = datetime.now(timezone.utc).isoformat()

    # --- Conversation State Management ---

    def add_conversation_turn(self, conversation_id: str, from_role: str, message: str, merchant_id: Optional[str] = None):
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            self._conversations[conversation_id].append({
                "from": from_role,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if merchant_id:
                if merchant_id not in self._merchant_conversations:
                    self._merchant_conversations[merchant_id] = []
                if conversation_id not in self._merchant_conversations[merchant_id]:
                    self._merchant_conversations[merchant_id].append(conversation_id)

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def has_sent_body_in_conversation(self, conversation_id: str, body: str) -> bool:
        normalized = body.strip().lower()
        with self._lock:
            hashes = self._sent_message_hashes.setdefault(conversation_id, set())
            return normalized in hashes

    def record_sent_body(self, conversation_id: str, body: str):
        normalized = body.strip().lower()
        with self._lock:
            hashes = self._sent_message_hashes.setdefault(conversation_id, set())
            hashes.add(normalized)

    def clear(self):
        with self._lock:
            self._contexts.clear()
            self._conversations.clear()
            self._merchant_conversations.clear()
            self._suppressed_keys.clear()
            self._opted_out_merchants.clear()
            self._sent_message_hashes.clear()


# Global Singleton Context Store
global_context_store = ContextStore()
