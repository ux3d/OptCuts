#include "GIF.hpp"
#include "IglUtils.hpp"

#include <igl/png/writePNG.h>
#include <igl/readOFF.h>

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>

#include "OptCuts.hpp"

int main(int argc, char* argv[])
{
    int progMode = 0;
    bool headlessMode = false;

    auto optCuts = OptCuts::OptCutsOptimization();

    if (argc > 1) {
        progMode = std::stoi(argv[1]);
    }
    switch (progMode) {
    case 0:
        // optimization mode
        std::cout << "Optimization mode" << std::endl;
        break;

    case 10: {
        // offline optimization mode
        std::cout << "Offline optimization mode" << std::endl;
        break;
    }

    case 100: {
        // headless mode
        optCuts.headlessMode = true;
        std::cout
            << "Headless mode" << std::endl;
        break;
    }

    default: {
        std::cout << "No progMode " << progMode << std::endl;
        return 0;
    }
    }

    // Optimization mode
    mkdir(optCuts.outputFolderPath.c_str(), 0777);

    std::string meshFileName("cone2.0.obj");
    if (argc > 2) {
        meshFileName = std::string(argv[2]);
    }
    std::string meshFilePath = meshFileName;
    meshFileName = meshFileName.substr(meshFileName.find_last_of('/') + 1);

    std::string meshFolderPath = meshFilePath.substr(0, meshFilePath.find_last_of('/'));
    std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));

    // Load mesh
    Eigen::MatrixXd V, UV, N;
    Eigen::MatrixXi F, FUV, FN;
    const std::string suffix = meshFilePath.substr(meshFilePath.find_last_of('.'));
    bool loadSucceed = false;
    if (suffix == ".off") {
        loadSucceed = igl::readOFF(meshFilePath, V, F);
    } else if (suffix == ".obj") {
        loadSucceed = igl::readOBJ(meshFilePath, V, UV, N, F, FUV, FN);
    } else {
        std::cout << "unkown mesh file format!" << std::endl;
        return -1;
    }

    if (!loadSucceed) {
        std::cout << "failed to load mesh!" << std::endl;
        return -1;
    }
    auto vertAmt_input = V.rows();

    Eigen::VectorXi B;
    bool isManifold = igl::is_vertex_manifold(F, B) && igl::is_edge_manifold(F);
    if (!isManifold) {
        std::cout << "input mesh contains non-manifold edges or vertices" << std::endl;
        std::cout << "please cleanup the mesh and retry" << std::endl;
        exit(-1);
    }

    const auto optimized = optCuts.optimize_mesh(V, UV, F, FUV);
    const auto& oV = std::get<0>(optimized);
    const auto& oUV = std::get<1>(optimized);
    const auto& oF = std::get<2>(optimized);
    const auto& oFUV = std::get<3>(optimized);

    // Output mesh as obj

    std::string outputFolderPath = optCuts.outputFolderPath + meshName + "/";
    mkdir(outputFolderPath.c_str(), 0777);

    igl::writeOBJ(outputFolderPath + "finalResult_mesh.obj", oV, oUV, N, oF, oFUV, FN);
}
