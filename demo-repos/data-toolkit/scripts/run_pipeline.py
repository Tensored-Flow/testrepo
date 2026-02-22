"""Main pipeline entry point. Category C -- skip."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.transforms import deduplicate_records, normalize_column
from toolkit.io_handlers import load_and_process_csv, save_results_to_db


def main():
    print("Loading data...")
    records = load_and_process_csv("data/input.csv")

    print(f"Loaded {len(records)} records")
    unique = deduplicate_records(records, ["id", "name"])
    print(f"After dedup: {len(unique)} records")

    if unique and "value" in unique[0]:
        values = [float(r["value"]) for r in unique]
        normalized = normalize_column(values)
        for i, record in enumerate(unique):
            record["normalized_value"] = normalized[i]

    save_results_to_db(unique, "output.db")
    print("Done!")


if __name__ == "__main__":
    main()
