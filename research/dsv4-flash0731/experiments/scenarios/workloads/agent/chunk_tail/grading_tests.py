import unittest
import tinyfix

class GradingTests(unittest.TestCase):
    def test_sizes(self):
        for n in range(10):
            for size in range(1, 12):
                values = list(range(n))
                result = tinyfix.chunked(values, size)
                self.assertEqual([x for part in result for x in part], values)
                self.assertTrue(all(0 < len(part) <= size for part in result))
                self.assertTrue(all(len(part) == size for part in result[:-1]))
                self.assertEqual(values, list(range(n)))

    def test_bad_size(self):
        for size in [0, -1, -10]:
            with self.assertRaises(ValueError):
                tinyfix.chunked([1], size)
