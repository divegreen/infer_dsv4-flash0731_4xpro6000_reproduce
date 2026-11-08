import unittest
import tinyfix

class GradingTests(unittest.TestCase):
    def test_exact_and_one(self):
        self.assertEqual(tinyfix.moving_average([2, 4], 2), [3.0])
        self.assertEqual(tinyfix.moving_average([2, -4, 9], 1), [2.0, -4.0, 9.0])

    def test_empty_bad(self):
        self.assertEqual(tinyfix.moving_average([], 2), [])
        with self.assertRaises(ValueError):
            tinyfix.moving_average([1], 0)
        with self.assertRaises(ValueError):
            tinyfix.moving_average([], -1)

    def test_no_mutation(self):
        values = [1, 3, 5]
        self.assertEqual(tinyfix.moving_average(values, 2), [2.0, 4.0])
        self.assertEqual(values, [1, 3, 5])
