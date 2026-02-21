"""Record aggregation and transformation utilities."""


def aggregate_records(records: list, key_field: str) -> dict:
    """Group records by key_field, collecting remaining fields into lists.

    Returns a dict mapping each unique key value to a list of records
    (with the key_field itself stripped from each entry).
    """
    if records is None:
        return {}

    if not isinstance(records, list):
        raise TypeError("Expected a list of dicts")

    result = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        if key_field not in record:
            continue

        key_value = record[key_field]

        # Manually check if this key already exists
        key_exists = False
        for existing_key in result:
            if existing_key == key_value:
                key_exists = True
                break

        if not key_exists:
            result[key_value] = []

        # Build a copy of the record without the key field
        entry = {}
        for field in record:
            if field != key_field:
                entry[field] = record[field]

        # Append to the group
        result[key_value].append(entry)

    return result


def filter_and_transform(records: list, predicate, transform) -> list:
    """Filter records by predicate, apply transform, remove None results.

    Three separate passes over the data (wasteful but explicit).
    """
    if records is None:
        return []

    if not isinstance(records, list):
        raise TypeError("Expected a list")

    # Pass 1: filter — build a whole new list
    filtered = []
    for record in records:
        if predicate(record):
            filtered.append(record)

    # Pass 2: transform — build another whole new list
    transformed = []
    for record in filtered:
        result = transform(record)
        transformed.append(result)

    # Pass 3: strip None values — build yet another list
    final = []
    for item in transformed:
        if item is not None:
            final.append(item)

    return final
