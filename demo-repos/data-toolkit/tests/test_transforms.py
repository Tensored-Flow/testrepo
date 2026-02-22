"""Tests for the transforms module.

These tests verify the correctness of data transformation functions.
The optimization pipeline should skip this file entirely (Category C).
"""

import unittest
from toolkit.transforms import deduplicate_records, normalize_column, flatten_nested_groups


class TestDeduplicateRecords(unittest.TestCase):

    def test_basic_dedup(self):
        records = [
            {"name": "Alice", "dept": "Eng"},
            {"name": "Bob", "dept": "Sales"},
            {"name": "Alice", "dept": "Eng"},
        ]
        result = deduplicate_records(records, ["name", "dept"])
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        result = deduplicate_records([], ["name"])
        self.assertEqual(result, [])

    def test_no_duplicates(self):
        records = [
            {"name": "Alice", "dept": "Eng"},
            {"name": "Bob", "dept": "Sales"},
        ]
        result = deduplicate_records(records, ["name"])
        self.assertEqual(len(result), 2)

    def test_case_insensitive(self):
        records = [
            {"name": "Alice"},
            {"name": "alice"},
            {"name": "ALICE"},
        ]
        result = deduplicate_records(records, ["name"])
        self.assertEqual(len(result), 1)


class TestNormalizeColumn(unittest.TestCase):

    def test_basic_normalize(self):
        result = normalize_column([0, 50, 100])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0)

    def test_empty(self):
        self.assertEqual(normalize_column([]), [])

    def test_single_value(self):
        result = normalize_column([42])
        self.assertEqual(result, [0.0])

    def test_all_same(self):
        result = normalize_column([5, 5, 5])
        self.assertEqual(result, [0.0, 0.0, 0.0])


class TestFlattenNestedGroups(unittest.TestCase):

    def test_basic_flatten(self):
        groups = {"A": [3, 1], "B": [2, 1]}
        result = flatten_nested_groups(groups)
        self.assertEqual(result, [1, 2, 3])

    def test_empty_groups(self):
        self.assertEqual(flatten_nested_groups({}), [])

    def test_single_group(self):
        result = flatten_nested_groups({"X": [5, 3, 1]})
        self.assertEqual(result, [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
