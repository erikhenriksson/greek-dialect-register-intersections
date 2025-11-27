import json
import statistics
from collections import defaultdict

# Store all dialect probs for each register
register_dialect_probs = defaultdict(lambda: defaultdict(list))

with open("labeled_sentences_chunk_1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        register = "-".join(data["registers"])

        for dialect, prob in data["dialect_probs"].items():
            register_dialect_probs[register][dialect].append(prob)

# Analyze distributions
for register in sorted(register_dialect_probs.keys()):
    print(f"\n{'=' * 80}")
    print(f"REGISTER: {register}")
    print(f"Total sentences: {len(register_dialect_probs[register]['standard'])}")
    print(f"{'=' * 80}")

    for dialect in sorted(register_dialect_probs[register].keys()):
        probs = register_dialect_probs[register][dialect]

        # Count how many exceed 0.5 threshold
        above_threshold = sum(1 for p in probs if p >= 0.5)
        pct_above = (above_threshold / len(probs)) * 100

        print(
            f"\n{dialect:15} | mean={statistics.mean(probs):.4f} | "
            f"median={statistics.median(probs):.4f} | "
            f"max={max(probs):.4f}"
        )
        print(f"                | >0.5 threshold: {above_threshold} ({pct_above:.1f}%)")

        # Show distribution of probabilities
        ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        dist = [sum(1 for p in probs if low <= p < high) for low, high in ranges]
        print(f"                | distribution: ", end="")
        for (low, high), count in zip(ranges, dist):
            if count > 0:
                print(f"[{low}-{high}): {count}  ", end="")
        print()
