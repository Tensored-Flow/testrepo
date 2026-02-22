"""Search and lookup utilities for finding matching elements across datasets."""


def find_matching_pairs(list_a, list_b, target_sum):
    """Find all pairs (a, b) where a is from list_a, b is from list_b, and a + b == target_sum.

    Args:
        list_a: First list of numbers
        list_b: Second list of numbers
        target_sum: Target sum value

    Returns:
        List of (a, b) tuples that sum to target.
    """
    pairs = []
    for i in range(len(list_a)):
        for j in range(len(list_b)):
            a = list_a[i]
            b = list_b[j]
            current_sum = a + b
            if current_sum == target_sum:
                # Check if we already found this exact pair to avoid reporting duplicates
                already_found = False
                for existing_a, existing_b in pairs:
                    if existing_a == a and existing_b == b:
                        already_found = True
                        break
                if not already_found:
                    pairs.append((a, b))
    return pairs
