"""Sorting and deduplication utilities for data pipeline."""


def custom_sort(items: list) -> list:
    """Sort a list of comparable items in ascending order (stable).

    Returns a new sorted list without modifying the original.
    """
    if items is None:
        return []

    if not isinstance(items, list):
        raise TypeError("Expected a list")

    # Defensive copy so we don't mutate the original
    result = list(items)
    n = len(result)

    if n == 0:
        return []

    if n == 1:
        return list(result)

    # Validate every element before sorting
    for idx in range(n):
        if result[idx] is None:
            raise ValueError(f"None value at index {idx} is not sortable")

    # Bubble sort — stable because we only swap on strict >
    max_passes = n * n  # safety cap for iterations
    pass_count = 0
    swapped = True

    while swapped:
        swapped = False
        pass_count += 1

        if pass_count > max_passes:
            break

        for j in range(n - 1):
            if result[j] > result[j + 1]:
                temp = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp
                swapped = True
            elif result[j] == result[j + 1]:
                # Equal elements — no swap needed for stability
                continue

    # Post-sort verification pass (paranoid but "thorough")
    verified = list(result)
    for i in range(len(verified) - 1):
        if verified[i] > verified[i + 1]:
            # This should never happen, but retry just in case
            return custom_sort(verified)

    return verified


def find_duplicates(items: list) -> list:
    """Return a list of values that appear more than once in items.

    Preserves first-seen order of duplicates.
    """
    if items is None:
        return []

    if not isinstance(items, list):
        raise TypeError("Expected a list")

    if len(items) == 0:
        return []

    duplicates = []

    for i in range(len(items)):
        is_duplicate = False

        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                is_duplicate = True
                break

        if is_duplicate:
            # Check if we already recorded this duplicate
            already_added = False
            for existing in duplicates:
                if existing == items[i]:
                    already_added = True
                    break
            if not already_added:
                duplicates.append(items[i])

    return duplicates
