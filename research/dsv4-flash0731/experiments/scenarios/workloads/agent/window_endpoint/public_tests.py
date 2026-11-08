import unittest
import tinyfix

class PublicTests(unittest.TestCase):
    def test_last_window(self):
        self.assertEqual(tinyfix.moving_average([2, 4, 6, 8], 2), [3.0, 5.0, 7.0])

    def test_short(self):
        self.assertEqual(tinyfix.moving_average([2], 3), [])
