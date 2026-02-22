"""Demo functions for testing the ComplexityImprover pipeline."""


# ═══ Category A: Pure, optimizable functions ═══

def bubble_sort(arr):
    """Classic O(n^2) bubble sort — should be flagged for optimization."""
    n = len(arr)
    result = arr[:]
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result


def find_duplicates(lst):
    """O(n^2) duplicate finder — could use a set for O(n)."""
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates


def matrix_multiply(a, b):
    """Naive O(n^3) matrix multiplication."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Incompatible matrices")
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def fibonacci_recursive(n):
    """Exponential fibonacci — classic optimization target."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def count_primes(n):
    """Count primes up to n using trial division — could use sieve."""
    count = 0
    for num in range(2, n + 1):
        is_prime = True
        for divisor in range(2, int(num ** 0.5) + 1):
            if num % divisor == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count


# ═══ Category B: Side effects (analysis only) ═══

def write_log(message, filename="output.log"):
    """Has file I/O — Category B."""
    import os
    with open(filename, "a") as f:
        f.write(f"{message}\n")
    return os.path.getsize(filename)


def fetch_data(url):
    """Network I/O — Category B."""
    import urllib.request
    response = urllib.request.urlopen(url)
    data = response.read()
    return len(data)


# ═══ Category C: Trivial / Skip ═══

def add(a, b):
    """Too simple to optimize."""
    return a + b


def get_name():
    """Trivial getter."""
    return "ComplexityImprover"
