import json
from collections import defaultdict

# Store examples for each register-dialect combo
examples = defaultdict(list)

with open("your_file.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        register = "-".join(data["registers"])

        for dialect, prob in data["dialect_probs"].items():
            if prob >= 0.5:
                # Store up to 5 examples per combo
                if len(examples[(register, dialect)]) < 5:
                    examples[(register, dialect)].append(
                        {
                            "text": data["text"],
                            "prob": prob,
                            "doc_id": data["doc_id"],
                            "sent_id": data["sent_id"],
                        }
                    )

# Print results
for (register, dialect), example_list in sorted(examples.items()):
    print(f"\n{'=' * 80}")
    print(f"Register: {register} | Dialect: {dialect} | Count: {len(example_list)}")
    print(f"{'=' * 80}")
    for i, ex in enumerate(example_list, 1):
        print(
            f"\n{i}. [prob={ex['prob']:.3f}, doc={ex['doc_id']}, sent={ex['sent_id']}]"
        )
        print(f"   {ex['text'][:200]}...")  # First 200 chars
