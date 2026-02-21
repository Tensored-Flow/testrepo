"""Tests for src/utils/sort.py — all must pass on the deliberately bad implementations."""

import random

from src.utils.sort import custom_sort, find_duplicates


class TestCustomSort:
    def test_empty_list(self):
        assert custom_sort([]) == []

    def test_single_element(self):
        assert custom_sort([42]) == [42]

    def test_already_sorted(self):
        assert custom_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        assert custom_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        assert custom_sort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]) == [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]

    def test_negative_numbers(self):
        assert custom_sort([-3, -1, -4, -1, -5]) == [-5, -4, -3, -1, -1]

    def test_mixed_types_integers(self):
        """Positive, negative, and zero."""
        assert custom_sort([0, -2, 5, -8, 3, 0, 1]) == [-8, -2, 0, 0, 1, 3, 5]

    def test_large_list(self):
        original = list(range(1000))
        shuffled = original[:]
        random.seed(42)
        random.shuffle(shuffled)
        assert custom_sort(shuffled) == original


class TestFindDuplicates:
    def test_find_duplicates_basic(self):
        result = find_duplicates([1, 2, 3, 2, 1, 4])
        assert set(result) == {1, 2}

    def test_find_duplicates_none(self):
        assert find_duplicates([1, 2, 3, 4, 5]) == []
