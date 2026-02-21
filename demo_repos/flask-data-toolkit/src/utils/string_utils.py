"""String matching and text analysis utilities."""


def fuzzy_match(s1: str, s2: str) -> int:
    """Compute the edit distance (Levenshtein distance) between two strings.

    Uses naive recursion — no memoization. Exponential time complexity.
    Only practical for short strings (len < 12 or so).
    """
    if s1 is None or s2 is None:
        return -1

    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("Expected strings")

    if len(s1) == 0:
        return len(s2)

    if len(s2) == 0:
        return len(s1)

    # If last characters match, no edit needed for this position
    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 1

    # Recursively compute all three operations
    insert_op = fuzzy_match(s1, s2[:-1]) + 1
    delete_op = fuzzy_match(s1[:-1], s2) + 1
    replace_op = fuzzy_match(s1[:-1], s2[:-1]) + cost

    # Manual min instead of min()
    best = insert_op
    if delete_op < best:
        best = delete_op
    if replace_op < best:
        best = replace_op

    return best


def count_word_frequencies(text: str) -> dict:
    """Count how many times each word appears in the text.

    Words are lowercased. Splits on whitespace manually.
    """
    if text is None:
        return {}

    if not isinstance(text, str):
        raise TypeError("Expected a string")

    if len(text) == 0:
        return {}

    # Manual whitespace splitting (instead of str.split())
    words = []
    current_word = ""

    for char in text:
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            if len(current_word) > 0:
                words.append(current_word.lower())
            current_word = ""
        else:
            current_word = current_word + char

    # Don't forget the last word
    if len(current_word) > 0:
        words.append(current_word.lower())

    # Manual counting (instead of collections.Counter)
    frequencies = {}

    for word in words:
        found = False
        for existing in frequencies:
            if existing == word:
                frequencies[word] = frequencies[word] + 1
                found = True
                break
        if not found:
            frequencies[word] = 1

    return frequencies
