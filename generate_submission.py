"""
generate_submission.py - Generate submission.jsonl for the 30 Benchmark Test Pairs
===================================================================================
Produces the pre-generated benchmark submissions complying with the exact Project Plan
schema (5.1) and challenge rubric.
"""

from __future__ import annotations
import json
from pathlib import Path
from composer import compose


def main():
    root = Path(__file__).parent
    dataset_dir = root / "dataset"
    expanded_dir = dataset_dir / "expanded"

    # Load categories
    categories = {}
    for cat_file in (dataset_dir / "categories").glob("*.json"):
        with open(cat_file) as fp:
            data = json.load(fp)
            categories[data["slug"]] = data

    # Load merchants
    merchants = {}
    for m_file in (expanded_dir / "merchants").glob("*.json"):
        with open(m_file) as fp:
            data = json.load(fp)
            merchants[data["merchant_id"]] = data

    # Load customers
    customers = {}
    for c_file in (expanded_dir / "customers").glob("*.json"):
        with open(c_file) as fp:
            data = json.load(fp)
            customers[data["customer_id"]] = data

    # Load triggers
    triggers = {}
    for t_file in (expanded_dir / "triggers").glob("*.json"):
        with open(t_file) as fp:
            data = json.load(fp)
            triggers[data["id"]] = data

    # Load test pairs
    with open(expanded_dir / "test_pairs.json") as fp:
        test_pairs = json.load(fp)["pairs"]

    print(f"Loaded {len(test_pairs)} test pairs. Generating benchmark submissions...")

    submission_lines = []
    for pair in test_pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        trigger = triggers.get(trigger_id, {})
        merchant = merchants.get(merchant_id, {})
        customer = customers.get(customer_id) if customer_id else None
        cat_slug = merchant.get("category_slug", trigger.get("payload", {}).get("category", "dentists"))
        category = categories.get(cat_slug, {})

        composed = compose(category, merchant, trigger, customer)

        # Build schema complying with Project Plan §5.1 & Challenge Brief §7.2
        entry = {
            "test_id": test_id,
            "category": cat_slug,
            "merchant_id": merchant_id,
            "trigger_type": trigger.get("kind", "generic"),
            "customer_id": customer_id,
            "composed_message": composed["body"],
            "body": composed["body"],
            "cta": composed.get("cta", "binary_yes_no"),
            "lever_used": composed.get("lever_used", "effort_externalization"),
            "facts_used": composed.get("facts_used", []),
            "language": composed.get("language", "en"),
            "send_as": composed.get("send_as", "vera"),
            "suppression_key": composed.get("suppression_key", ""),
            "rationale": composed.get("rationale", "")
        }
        submission_lines.append(json.dumps(entry, ensure_ascii=False))

    output_path = root / "submission.jsonl"
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(submission_lines) + "\n")

    json_output_path = root / "submissions.json"
    with open(json_output_path, "w", encoding="utf-8") as fp:
        json.dump([json.loads(line) for line in submission_lines], fp, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(submission_lines)} lines in {output_path} and {json_output_path}")


if __name__ == "__main__":
    main()
