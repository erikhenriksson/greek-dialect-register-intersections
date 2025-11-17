import json
from collections import Counter

register_dialect_counts = Counter()

with open("your_file.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        register = "-".join(data["registers"])

        for dialect, prob in data["dialect_probs"].items():
            if prob >= 0.5:
                register_dialect_counts[(register, dialect)] += 1

# Print results
for (register, dialect), count in sorted(register_dialect_counts.items()):
    print(f"{register}\t{dialect}\t{count}")
