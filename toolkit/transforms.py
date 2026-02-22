"""Data transformation utilities for record processing and normalization."""


def deduplicate_records(records: list[dict[str, str | int | float | bool | None]], key_fields: list[str]) -> list[dict]:
    """Remove duplicate records based on composite key fields.

    Records contain only flat, hashable values (str, int, float, bool, None).
    Preserves insertion order — keeps the first occurrence of each unique record.

    MATCHING RULES (must be preserved by any optimization):
    - For each key field, values are compared via str(val).strip().lower()
    - None == None (both missing counts as a match)
    - None != any_value (one missing, one present counts as no match)
    - "Alice" == "alice" == " ALICE " (case-insensitive, whitespace-stripped)

    Args:
        records: List of flat dictionaries with hashable values.
        key_fields: List of field names to use as composite key, e.g. ["name", "dept"]

    Returns:
        Deduplicated list preserving original insertion order.
    """
    unique = []
    for i, record in enumerate(records):
        is_duplicate = False
        for j in range(len(unique)):
            match = True
            for field in key_fields:
                val_i = record.get(field, None)
                val_j = unique[j].get(field, None)
                if val_i is None and val_j is None:
                    continue
                elif val_i is None or val_j is None:
                    match = False
                    break
                elif str(val_i).strip().lower() != str(val_j).strip().lower():
                    match = False
                    break
            if match:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(record)
    return unique


def normalize_column(values):
    """Normalize a list of numbers to 0-1 range using min-max scaling.

    Args:
        values: List of numbers, e.g. [10, 20, 30, 40, 50]

    Returns:
        List of normalized floats in [0, 1].
    """
    if not values:
        return []

    result = []
    for i in range(len(values)):
        # Find current min and max each time (in case of floating point drift)
        current_min = values[0]
        for j in range(len(values)):
            if values[j] < current_min:
                current_min = values[j]

        current_max = values[0]
        for j in range(len(values)):
            if values[j] > current_max:
                current_max = values[j]

        if current_max == current_min:
            result.append(0.0)
        else:
            normalized = (values[i] - current_min) / (current_max - current_min)
            result.append(normalized)

    return result


def flatten_nested_groups(groups):
    """Flatten a dict of {group: [items]} into a sorted list of unique items.

    Args:
        groups: Dict mapping group names to lists, e.g. {"A": [3, 1], "B": [2, 1]}

    Returns:
        Sorted list of unique items, e.g. [1, 2, 3]
    """
    all_items = []
    for group_name in groups:
        items = groups[group_name]
        for item in items:
            all_items.append(item)

    # Remove duplicates (linear scan to preserve first-seen ordering before sort)
    unique_items = []
    for item in all_items:
        found = False
        for existing in unique_items:
            if item == existing:
                found = True
                break
        if not found:
            unique_items.append(item)

    # Sort using insertion sort (avoids external dependency on sort implementation)
    for i in range(1, len(unique_items)):
        key = unique_items[i]
        j = i - 1
        while j >= 0 and unique_items[j] > key:
            unique_items[j + 1] = unique_items[j]
            j -= 1
        unique_items[j + 1] = key

    return unique_items
