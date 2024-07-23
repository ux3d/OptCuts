import unittest
import igl
import PyOptCuts

class TestPyOptCuts(unittest.TestCase):
    def test_optimize(self):
        vertices, uvs, normals, faces, uv_indices, normal_indices = igl.read_obj("tests/bimba.obj")
        (vertices, uvs, faces, uv_indices) = PyOptCuts.optimize(vertices, uvs, faces, uv_indices)
        with open("tests/bimba_opt.obj", 'w') as file:
            for v in vertices:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for vt in uvs:
                file.write(f"vt {vt[0]} {vt[1]}\n")
            for f, fuv in zip(faces, uv_indices):
                file.write(f"f {f[0]+1}/{fuv[0]+1} {f[1]+1}/{fuv[1] + 1} {f[2]+1}/{fuv[2]+1}\n")

if __name__ == '__main__':
    unittest.main()
