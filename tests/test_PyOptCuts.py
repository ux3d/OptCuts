import unittest
import PyOptCuts

class TestPyOptCuts(unittest.TestCase):
    def test_optimize(self):
        self.assertEqual(PyOptCuts.optimize(), 'expected result')

if __name__ == '__main__':
    unittest.main()