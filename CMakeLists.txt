cmake_minimum_required(VERSION 3.9)
project(OptCuts)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

set(OPTCUTS_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(${PROJECT_NAME}Dependencies)

option(OPTCUTS_PYTHON "Build for python" ON)

# libigl
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)
if (OPTCUTS_PYTHON)
  option(LIBIGL_WITH_PNG               "Use STB_image"      OFF)
else()
  option(LIBIGL_WITH_PNG               "Use STB_image"      ON)
endif()

find_package(LIBIGL REQUIRED QUIET)


# Python Bindings
if (OPTCUTS_PYTHON)
  if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
  else()
    set(DEV_MODULE Development.Module)
  endif()
  
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter ${DEV_MODULE} OPTIONAL_COMPONENTS Development.SABIModule)
  
  if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()
  
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)
endif()

# Intel TBB
#include(ext/tbb/cmake/TBBGet.cmake)
#tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
#find_package(TBB REQUIRED tbb)


file(GLOB SRCFILES
  src/Optimizer.cpp
  src/TriMesh.cpp
  src/Scaffold.cpp

  src/Energy/Energy.cpp
  src/Energy/ARAPEnergy.cpp
  src/Energy/SymDirichletEnergy.cpp
  
  src/Utils/IglUtils.cpp
  
  src/LinSysSolver/EigenLibSolver.cpp
)

include_directories(
  src
  src/Energy
  src/Utils
  src/LinSysSolver
)

if (OPTCUTS_PYTHON)
  if (NOT SKBUILD)
    message(WARNING "\
    This CMake file is meant to be executed using 'scikit-build'. Running
    it directly will almost certainly not produce the desired result. If
    you are a user trying to install this package, please use the command
    below, which will install all necessary build dependencies, compile
    the package in an isolated environment, and then install it.
    =====================================================================
    $ pip install .
    =====================================================================
    If you are a software developer, and this is your own package, then
    it is usually much more efficient to install the build dependencies
    in your environment once and use the following command that avoids
    a costly creation of a new virtual environment at every compilation:
    =====================================================================
    $ pip install nanobind scikit-build-core[pyproject]
    $ pip install --no-build-isolation -ve .
    =====================================================================
    You may optionally add -Ceditable.rebuild=true to auto-rebuild when
    the package is imported. Otherwise, you need to re-run the above
    after editing C++ files.")
  endif()

  nanobind_add_module(_PyOptCutsImpl STABLE_ABI NB_STATIC src/PyOptCuts.cpp ${SRCFILES})
  target_link_libraries(_PyOptCutsImpl PRIVATE igl::core igl::opengl_glfw igl::triangle TBB::tbb)
  install(TARGETS _PyOptCutsImpl DESTINATION PyOptCuts)


else()
  add_executable(${PROJECT_NAME}_bin src/main.cpp ${SRCFILES})
  target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle igl::png TBB::tbb)
endif()

