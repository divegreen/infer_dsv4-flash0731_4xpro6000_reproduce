import unittest
import tinyfix

class PublicTests(unittest.TestCase):
    def test_right_only(self):
        self.assertEqual(tinyfix.merge_counts({"a": 2}, {"a": 3, "b": 4}), {"a": 5, "b": 4})

    def test_empty(self):
        self.assertEqual(tinyfix.merge_counts({}, {}), {})
