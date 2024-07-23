import unittest
import igl
import PyOptCuts

class TestPyOptCuts(unittest.TestCase):
    def test_optimize(self):
        v, tc, n, f, ftc, fn = igl.read_obj("tests/bimba.obj")
        self.assertEqual(PyOptCuts.optimize(v, tc, n, f, ftc, fn), 0)

if __name__ == '__main__':
    unittest.main()
