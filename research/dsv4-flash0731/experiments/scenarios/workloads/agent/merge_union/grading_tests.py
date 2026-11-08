import unittest
import tinyfix

class GradingTests(unittest.TestCase):
    def test_negative_zero(self):
        self.assertEqual(tinyfix.merge_counts({"x": -2, "z": 0}, {"x": 2, "y": -3}), {"x": 0, "z": 0, "y": -3})

    def test_no_mutation(self):
        left, right = {"a": 1}, {"b": 2}
        result = tinyfix.merge_counts(left, right)
        self.assertEqual(result, {"a": 1, "b": 2})
        self.assertEqual(left, {"a": 1})
        self.assertEqual(right, {"b": 2})
        self.assertIsNot(result, left)
        self.assertIsNot(result, right)

    def test_one_side_empty(self):
        self.assertEqual(tinyfix.merge_counts({}, {"b": 0}), {"b": 0})
