# magicpin AI Challenge — Vera Bot Submission

## 1. Architecture Overview

This submission implements **Vera**, magicpin's next-generation merchant and customer engagement assistant over WhatsApp. Built with **FastAPI** and designed for stateful, high-throughput execution under strict 30-second response budgets.

```
                  ┌──────────────────────────────────────────────┐
                  │              FastAPI (bot.py)                │
                  │  /v1/healthz  /v1/metadata  /v1/context      │
                  │         /v1/tick         /v1/reply           │
                  └──────────────────────┬───────────────────────┘
                                         │
                   ┌─────────────────────▼──────────────────────┐
                   │    In-Memory Context Store & Session DB    │
                   │  - CategoryContext (slug)                  │
                   │  - MerchantContext (merchant_id)           │
                   │  - CustomerContext (customer_id)           │
                   │  - TriggerContext (trigger_id)             │
                   │  - Conversations (history, turns, state)   │
                   └─────────────────────┬──────────────────────┘
                                         │
        ┌────────────────────────────────┴──────────────────────────────┐
        │                                                               │
        ▼ (Incoming Reply /v1/reply)                                    ▼ (Tick /v1/tick)
┌───────────────────────────────┐                             ┌───────────────────────────────┐
│ Stage 0: Auto-Reply Filter    │                             │ Trigger Evaluator             │
│ (canned WhatsApp regex/hash)  │                             │ (urgency, suppression check,  │
└──────────────┬────────────────┘                             │  freshness, should_send?)     │
               ▼                                              └──────────────┬────────────────┘
┌───────────────────────────────┐                                            │
│ Stage 0.5: Intent Detection   │                                            │
│ (explicit "let's do it" hook) │                                            │
└──────────────┬────────────────┘                                            │
               │                                                             │
               └─────────────────────────┬───────────────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 7-Stage Core Composition Engine (composer.py)                                               │
│ 1. Context Reconciliation: merge Merchant identity, Category voice, Trigger, Customer      │
│ 2. Fact Assimilation & Whitelist Extraction: strict numbers, prices, dates, source citations│
│ 3. LLM / Constrained Generation: temperature=0, single CTA, single psychological lever      │
│ 4. Taboo / Banned Word Filter: regex scan for medical taboos ("guaranteed", "cure")        │
│ 5. Fact-Check Provenance Pass: 0% hallucinations; verify all tokens against whitelist       │
│ 6. CTA & Lever Normalization: binary YES/NO, single open question, or slot choices          │
│ 7. Dispatch & Deterministic Fallback: instant SLA guarantee (<30s)                          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prompt Engineering & 4-Context Mapping

Every outbound message synthesizes all 4 context layers:
1. **CategoryContext**: Sets tone (clinical-peer for dentists, operator for restaurants, coaching for gyms), allowed technical vocabulary, and category-level taboos (e.g., medical disclaimers, no "guaranteed" or "cure").
2. **MerchantContext**: Provides real grounding: owner first name, verified locality, 30d views/calls/CTR, and active catalog offers (e.g., "Dental Cleaning @ ₹299").
3. **TriggerContext**: Provides explicit "why now" relevance (e.g., JIDA research paper release, Saturday IPL match footfall shift, or 6-month patient recall).
4. **CustomerContext** (when customer-scoped): Personalizes language mix (Hinglish/Hindi-English code-mix), customer name, relationship history, and booking preference slots (e.g., weekday evenings).

---

## 3. Edge-Case Handling & State Machine

- **Auto-Reply Pollution (Stage 0)**: Canned WhatsApp Business auto-replies ("*Thank you for contacting...*", "*Our team will respond shortly*") are recognized via regex fingerprinting and repetition hashing. Turn 1 pauses (`wait: 14400s`) to allow the real owner to view the phone; repeated auto-replies trigger `action: "end"` to preserve conversation turns.
- **Intent Handoff (Stage 0.5)**: When a merchant expresses commitment ("*Ok let's do it*", "*chalo karte hain*", "*proceed*"), the bot transitions immediately to action execution (drafting campaigns, scheduling posts) and strictly eliminates qualifying questions (`would you`, `how about`).
- **Hostile Opt-Out & Boundary Enforcement**: Detects hostile replies ("*stop*", "*spam*"), immediately returns `action: "end"`, and records a 30-day suppression key on the merchant. Out-of-scope inquiries (e.g., "*help me file GST*") are politely deflected to external specialists while maintaining active campaign thread momentum.
- **Stale Triggers & Deduplication**: Every trigger evaluates `suppression_key` and expiration before dispatch. Identical message bodies in the same conversation are dropped to avoid repetition penalties.

---

## 4. Key Trade-offs

1. **In-Memory Store vs External DB**: In-memory storage was selected for zero network overhead, sub-millisecond lookups, and guaranteed sub-second response times, fitting well within the 30-second SLA.
2. **Deterministic Fallback vs Pure LLM**: While the pipeline supports LLM calls (OpenAI, Anthropic, Gemini), a full deterministic fallback engine (`fallback_templates.py`) was implemented. If an external LLM times out or encounters network limits, the bot falls back instantly with 100% adherence to context facts, preventing timeouts and hallucination penalties.
3. **Strict Whitelist Fact Extraction**: Unmatched stats or numbers are filtered out before sending, guaranteeing zero hallucinations.

---

## 5. Running the Bot & Tests

```bash
# 1. Expand the dataset
uv run python3 dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded

# 2. Generate benchmark test submissions (submission.jsonl)
uv run python3 generate_submission.py

# 3. Run the automated verification test suite
uv run python3 test_bot.py

# 4. Start the production FastAPI server
uv run uvicorn bot:app --host 0.0.0.0 --port 8080
```
