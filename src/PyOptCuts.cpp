#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "OptCuts.hpp"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> optcuts_optimize(Eigen::MatrixXd& vertices, Eigen::MatrixXd& uvs, Eigen::MatrixXi& indices, Eigen::MatrixXi& uv_indices)
{
    OptCuts::OptCutsOptimization optcuts;
    return optcuts.optimize_mesh(vertices, uvs, indices, uv_indices);
}

NB_MODULE(_PyOptCutsImpl, m)
{
    m.def("optimize", &optcuts_optimize, "Optimize the mesh data");
}
