#pragma once

#include "Timer.hpp"
#include "Types.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace OptCuts {
struct OptimizationState {
    std::vector<std::pair<double, double>> energyChanges_bSplit, energyChanges_iSplit, energyChanges_merge;
    std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
    std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit, newVertPoses_merge;

    double filterExp_in = 0.6;
    int inSplitTotalAmt;

    OptCuts::MethodType methodType;
    bool fractureMode = false;

    Timer timer, timer_step;
};
}
