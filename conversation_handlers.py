"""
conversation_handlers.py - State Machine for Turn Handling
==========================================================
Implements:
- Stage 0: Auto-Reply Detection & Filtering
- Stage 0.5: Intent Detection & Action Fast-Path Transition
- Hostile & Opt-Out Handling
- Out-of-Scope / Curveball Redirection
- Multi-turn Follow-up Composition
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional, Tuple
from context_store import ContextStore, global_context_store


# Known WhatsApp Business canned / auto-reply patterns
AUTO_REPLY_PATTERNS = [
    r"thank\s+you\s+for\s+contacting",
    r"thanks\s+for\s+contacting",
    r"our\s+team\s+will\s+respond\s+shortly",
    r"will\s+respond\s+shortly",
    r"automated\s+assistant",
    r"auto[-\s]?reply",
    r"currently\s+away",
    r"we\s+are\s+closed",
    r"reach\s+us\s+during\s+business\s+hours",
    r"aapki\s+jaankari\s+ke\s+liye\s+bahut\s+shukriya",
    r"hamari\s+team\s+tak\s+pahuncha\s+deti\s+hoon",
]

# Explicit commitment / action transition patterns
INTENT_COMMIT_PATTERNS = [
    r"\blet'?s\s+do\s+it\b",
    r"\bwhat'?s\s+next\b",
    r"\bok\s+lets\s+do\s+it\b",
    r"\byes\s+send\b",
    r"\bgo\s+ahead\b",
    r"\bproceed\b",
    r"\bi\s+want\s+to\s+join\b",
    r"\bchalo\s+karte\s+hain\b",
    r"\bhaan\s+bhejo\b",
    r"\bmujhe\s+judna\s+hai\b",
    r"\bconfirm\b",
    r"\bsign\s+me\s+up\b",
    r"\byes\s+please\b",
    r"\byes,\s+let'?s\s+do\s+it\b"
]

# Hostility / opt-out patterns
HOSTILE_OPT_OUT_PATTERNS = [
    r"\bstop\s+messaging\b",
    r"\buseless\s+spam\b",
    r"\bstop\s+sending\b",
    r"\bnot\s+interested\b",
    r"\bdon'?t\s+message\b",
    r"\bunsubscribe\b",
    r"\bspam\b",
    r"\bbothering\s+me\b",
    r"\bleave\s+me\s+alone\b",
    r"\bmat\s+bhejo\b",
    r"\broko\b"
]

# Out-of-scope / curveball patterns
OUT_OF_SCOPE_PATTERNS = [
    (r"\bgst\b", "GST filing and accounting are best handled by your CA/accountant"),
    (r"\btax\b", "Tax advisory is outside what I can assist with directly"),
    (r"\bloan\b", "Business loans and financing are outside what I manage"),
    (r"\blegal\b", "Legal advisory is outside my scope"),
]


class ConversationHandler:
    def __init__(self, store: Optional[ContextStore] = None):
        self.store = store or global_context_store
        # Track auto-reply frequency per merchant/conversation
        self._auto_reply_counts: Dict[str, int] = {}

    def is_auto_reply(self, message: str) -> bool:
        """Stage 0: Fast deterministic check against known auto-reply phrasing."""
        normalized = message.strip().lower()
        for pattern in AUTO_REPLY_PATTERNS:
            if re.search(pattern, normalized):
                return True
        return False

    def is_hostile_or_opt_out(self, message: str) -> bool:
        normalized = message.strip().lower()
        for pattern in HOSTILE_OPT_OUT_PATTERNS:
            if re.search(pattern, normalized):
                return True
        return False

    def is_intent_commitment(self, message: str) -> bool:
        """Stage 0.5: Check for explicit affirmative action intent."""
        normalized = message.strip().lower()
        for pattern in INTENT_COMMIT_PATTERNS:
            if re.search(pattern, normalized):
                return True
        return False

    def check_out_of_scope(self, message: str) -> Optional[str]:
        normalized = message.strip().lower()
        for pattern, explanation in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, normalized):
                return explanation
        return None

    def handle_reply(self, conversation_id: str, merchant_id: Optional[str], customer_id: Optional[str], from_role: str, message: str, turn_number: int) -> Dict[str, Any]:
        """
        Processes incoming reply across all lifecycle states.
        Returns the action dict: {"action": "send" | "wait" | "end", ...}
        """
        # Record turn in conversation store
        self.store.add_conversation_turn(conversation_id, from_role, message, merchant_id)

        # 1. Check for Hostility / Opt-Out
        if self.is_hostile_or_opt_out(message):
            if merchant_id:
                self.store.opt_out_merchant(merchant_id)
            return {
                "action": "end",
                "rationale": "Merchant explicitly opted out / expressed hostility. Closing conversation immediately and suppressing future triggers."
            }

        # 2. Stage 0: Auto-Reply Detection
        track_key = merchant_id or conversation_id
        if self.is_auto_reply(message):
            count = self._auto_reply_counts.get(track_key, 0) + 1
            self._auto_reply_counts[track_key] = count

            if count == 1 and turn_number <= 2:
                # First auto-reply: Wait for owner to see real device
                return {
                    "action": "wait",
                    "wait_seconds": 14400,
                    "rationale": "Detected merchant auto-reply (canned 'Thank you for contacting' phrasing). Backing off 4 hours to wait for owner."
                }
            else:
                # Repeated auto-reply: End to prevent turn pollution
                return {
                    "action": "end",
                    "rationale": f"Detected repeated auto-reply ({count} times) with zero real engagement. Gracefully closing."
                }

        # 3. Stage 0.5: Intent Detection & Fast-Path Action Execution
        if self.is_intent_commitment(message):
            # Resolve merchant context if available
            merchant = self.store.get_merchant(merchant_id) if merchant_id else None
            m_name = merchant.get("identity", {}).get("owner_first_name") if merchant else None
            salutation = f"{m_name}, " if m_name else ""

            # Check if there is an active offer or trigger to reference
            active_offer = ""
            if merchant and merchant.get("offers"):
                for off in merchant["offers"]:
                    if off.get("status") == "active":
                        active_offer = off.get("title", "")
                        break

            offer_mention = f" for '{active_offer}'" if active_offer else ""
            body = (
                f"Done! {salutation}Drafting your WhatsApp campaign{offer_mention} now — 90 seconds. "
                "I will also prepare the scheduled Google post for tomorrow 10am. "
                "Reply CONFIRM to proceed with the next step."
            )
            # Ensure NO qualifying words exist: ["would you", "do you", "can you tell", "what if", "how about"]
            return {
                "action": "send",
                "body": body,
                "cta": "binary_confirm_cancel",
                "rationale": "Merchant explicitly committed; switching immediately from qualification to action-execution with concrete next step."
            }

        # 4. Out-of-Scope / Curveball Redirection
        out_of_scope_note = self.check_out_of_scope(message)
        if out_of_scope_note:
            return {
                "action": "send",
                "body": (
                    f"I'll have to leave that to your specialist — {out_of_scope_note}. "
                    "Coming back to our active campaign — want me to draft the next post now?"
                ),
                "cta": "open_ended",
                "rationale": "Out-of-scope ask politely declined; redirects back to the original workflow without losing momentum."
            }

        # 5. Normal Conversational Turn (Engaged Follow-up)
        # Check merchant details
        merchant = self.store.get_merchant(merchant_id) if merchant_id else None
        owner_name = merchant.get("identity", {}).get("owner_first_name", "") if merchant else ""
        greeting = f"Sure {owner_name}, " if owner_name else "Got it! "

        body = (
            f"{greeting}here is the next step drafted for you. "
            "I have prepared the update and ready to push live. "
            "Reply YES to confirm and publish now."
        )
        return {
            "action": "send",
            "body": body,
            "cta": "binary_yes_no",
            "rationale": "Acknowledged merchant input and progressed conversation with low-friction binary action."
        }


# Global Singleton Handler
global_conversation_handler = ConversationHandler()
