import unittest
import tinyfix

class PublicTests(unittest.TestCase):
    def test_partial(self):
        self.assertEqual(tinyfix.chunked([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_empty(self):
        self.assertEqual(tinyfix.chunked([], 3), [])
