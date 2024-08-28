
#pragma once

#include <Eigen/Dense>

#include "GIF.hpp"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"
#include "Timer.hpp"
#include "Types.hpp"

#include "OptimizationState.hpp"

#include "cut_to_disk.hpp" // hasn't been pulled into the older version of libigl we use
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/boundary_loop.h>
#include <igl/cut_mesh.h>
#include <igl/edge_lengths.h>
#include <igl/euler_characteristic.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>

#include <igl/facet_components.h>

#include <sys/stat.h> // for mkdir

#include <ctime>
#include <fstream>
#include <string>
#include <tuple>

namespace OptCuts {
struct OptCutsOptimization {
    OptCuts::OptimizationState optState = {};
    // optimization

    std::vector<const OptCuts::TriMesh*> triSoup;
    OptCuts::TriMesh triSoup_backup = OptCuts::TriMesh(optState);
    OptCuts::Optimizer* optimizer;
    std::vector<OptCuts::Energy*> energyTerms;
    std::vector<double> energyParams;

    bool bijectiveParam = true;
    bool rand1PInitCut = false;
    double lambda_init;
    int iterNum = 0;
    int converged = 0;
    double fracThres = 0.0;
    bool topoLineSearch = true;
    int initCutOption = 0;
    double upperBound = 4.1;
    const double convTol_upperBound = 1.0e-3;

    int opType_queried = -1;
    std::vector<int> path_queried;
    Eigen::MatrixXd newVertPos_queried;
    bool reQuery = false;

    std::string outputFolderPath = "output/";

    // visualization
    bool headlessMode = true;
    static constexpr int channel_initial = 0;
    static constexpr int channel_result = 1;
    static constexpr int channel_findExtrema = 2;
    double texScale = 1.0;
    float fracTailSize = 15.0f;
    std::string infoName = "";

    void proceedOptimization(int proceedNum = 1)
    {
        for (int proceedI = 0; (proceedI < proceedNum) && (!converged); proceedI++) {
            std::cout << "Iteration" << iterNum << ":" << std::endl;
            converged = optimizer->solve(1);
            iterNum = optimizer->getIterNum();
        }
    }

    int computeOptPicked(const std::vector<std::pair<double, double>>& energyChanges0,
        const std::vector<std::pair<double, double>>& energyChanges1,
        double lambda)
    {
        assert(!energyChanges0.empty());
        assert(!energyChanges1.empty());
        assert((lambda >= 0.0) && (lambda <= 1.0));

        double minEChange0 = __DBL_MAX__;
        for (int ecI = 0; ecI < energyChanges0.size(); ecI++) {
            if ((energyChanges0[ecI].first == __DBL_MAX__) || (energyChanges0[ecI].second == __DBL_MAX__)) {
                continue;
            }
            double EwChange = energyChanges0[ecI].first * (1.0 - lambda) + energyChanges0[ecI].second * lambda;
            if (EwChange < minEChange0) {
                minEChange0 = EwChange;
            }
        }

        double minEChange1 = __DBL_MAX__;
        for (int ecI = 0; ecI < energyChanges1.size(); ecI++) {
            if ((energyChanges1[ecI].first == __DBL_MAX__) || (energyChanges1[ecI].second == __DBL_MAX__)) {
                continue;
            }
            double EwChange = energyChanges1[ecI].first * (1.0 - lambda) + energyChanges1[ecI].second * lambda;
            if (EwChange < minEChange1) {
                minEChange1 = EwChange;
            }
        }

        assert((minEChange0 != __DBL_MAX__) || (minEChange1 != __DBL_MAX__));
        return (minEChange0 > minEChange1);
    }

    int computeBestCand(const std::vector<std::pair<double, double>>& energyChanges, double lambda,
        double& bestEChange)
    {
        assert((lambda >= 0.0) && (lambda <= 1.0));

        bestEChange = __DBL_MAX__;
        int id_minEChange = -1;
        for (int ecI = 0; ecI < energyChanges.size(); ecI++) {
            if ((energyChanges[ecI].first == __DBL_MAX__) || (energyChanges[ecI].second == __DBL_MAX__)) {
                continue;
            }
            double EwChange = energyChanges[ecI].first * (1.0 - lambda) + energyChanges[ecI].second * lambda;
            if (EwChange < bestEChange) {
                bestEChange = EwChange;
                id_minEChange = ecI;
            }
        }

        return id_minEChange;
    }

    bool checkCand(const std::vector<std::pair<double, double>>& energyChanges)
    {
        for (const auto& candI : energyChanges) {
            if ((candI.first < 0.0) || (candI.second < 0.0)) {
                return true;
            }
        }

        double minEChange = __DBL_MAX__;
        for (const auto& candI : energyChanges) {
            if (candI.first < minEChange) {
                minEChange = candI.first;
            }
            if (candI.second < minEChange) {
                minEChange = candI.second;
            }
        }
        std::cout << "candidates not valid, minEChange: " << minEChange << std::endl;
        return false;
    }

    double updateLambda(double measure_bound, double lambda_SD, double kappa = 1.0, double kappa2 = 1.0)
    {
        lambda_SD = std::max(0.0, kappa * (measure_bound - (upperBound - convTol_upperBound / 2.0)) + kappa2 * lambda_SD / (1.0 - lambda_SD));
        return lambda_SD / (1.0 + lambda_SD);
    }

    bool updateLambda_stationaryV(bool cancelMomentum = true, bool checkConvergence = false)
    {
        Eigen::MatrixXd edgeLengths;
        igl::edge_lengths(triSoup[channel_result]->V_rest, triSoup[channel_result]->F, edgeLengths);
        const double eps_E_se = 1.0e-3 * edgeLengths.minCoeff() / triSoup[channel_result]->virtualRadius;

        // measurement and energy value computation
        const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
        double E_se;
        triSoup[channel_result]->computeSeamSparsity(E_se);
        E_se /= triSoup[channel_result]->virtualRadius;
        double stretch_l2, stretch_inf, stretch_shear, compress_inf;
        triSoup[channel_result]->computeStandardStretch(stretch_l2, stretch_inf, stretch_shear, compress_inf);
        double measure_bound = E_SD;
        const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound, energyParams[0]) - energyParams[0]));

        // TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated
        //  oscillation detection
        static int iterNum_bestFeasible = -1;
        static OptCuts::TriMesh triSoup_bestFeasible(optState);
        static double E_se_bestFeasible = __DBL_MAX__;
        static int lastStationaryIterNum = 0; // still necessary because boundary and interior query are with same iterNum
        static std::map<double, std::vector<std::pair<double, double>>> configs_stationaryV;
        if (iterNum != lastStationaryIterNum) {
            // not a roll back config
            const double lambda = 1.0 - energyParams[0];
            bool oscillate = false;
            const auto low = configs_stationaryV.lower_bound(E_se);
            if (low == configs_stationaryV.end()) {
                // all less than E_se
                if (!configs_stationaryV.empty()) {
                    // use largest element
                    if (std::abs(configs_stationaryV.rbegin()->first - E_se) < eps_E_se) {
                        for (const auto& lambdaI : configs_stationaryV.rbegin()->second) {
                            if ((std::abs(lambdaI.first - lambda) < eps_lambda) && (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                                oscillate = true;
                                std::cout << configs_stationaryV.rbegin()->first << ", " << lambdaI.second << std::endl;
                                std::cout << E_se << ", " << lambda << ", " << E_SD << std::endl;
                                break;
                            }
                        }
                    }
                }
            } else if (low == configs_stationaryV.begin()) {
                // all not less than E_se
                if (std::abs(low->first - E_se) < eps_E_se) {
                    for (const auto& lambdaI : low->second) {
                        if ((std::abs(lambdaI.first - lambda) < eps_lambda) && (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                            oscillate = true;
                            std::cout << low->first << ", " << lambdaI.first << ", " << lambdaI.second << std::endl;
                            std::cout << E_se << ", " << lambda << ", " << E_SD << std::endl;
                            break;
                        }
                    }
                }

            } else {
                const auto prev = std::prev(low);
                if (std::abs(low->first - E_se) < eps_E_se) {
                    for (const auto& lambdaI : low->second) {
                        if ((std::abs(lambdaI.first - lambda) < eps_lambda) && (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                            oscillate = true;
                            std::cout << low->first << ", " << lambdaI.first << ", " << lambdaI.second << std::endl;
                            std::cout << E_se << ", " << lambda << ", " << E_SD << std::endl;
                            break;
                        }
                    }
                }
                if ((!oscillate) && (std::abs(prev->first - E_se) < eps_E_se)) {
                    for (const auto& lambdaI : prev->second) {
                        if ((std::abs(lambdaI.first - lambda) < eps_lambda) && (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                            oscillate = true;
                            std::cout << prev->first << ", " << lambdaI.first << ", " << lambdaI.second << std::endl;
                            std::cout << E_se << ", " << lambda << ", " << E_SD << std::endl;
                            break;
                        }
                    }
                }
            }

            // record best feasible UV map
            if ((measure_bound <= upperBound) && (E_se < E_se_bestFeasible)) {
                iterNum_bestFeasible = iterNum;
                triSoup_bestFeasible = *triSoup[channel_result];
                E_se_bestFeasible = E_se;
            }

            if (oscillate && (iterNum_bestFeasible >= 0)) {
                // arrive at the best feasible config again
                std::cout << "oscillation detected at measure = " << measure_bound << ", b = " << upperBound << "lambda = " << energyParams[0] << std::endl;
                std::cout << lastStationaryIterNum << ", " << iterNum << std::endl;
                if (iterNum_bestFeasible != iterNum) {
                    optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                    std::cout << "rolled back to best feasible in iter " << iterNum_bestFeasible << std::endl;
                }
                return false;
            } else {
                configs_stationaryV[E_se].emplace_back(std::pair<double, double>(lambda, E_SD));
            }
        }
        lastStationaryIterNum = iterNum;

        // convergence check
        if (checkConvergence) {
            if (measure_bound <= upperBound) {
                // save info at first feasible stationaryVT for comparison
                static bool saved = false;
                if (!saved) {
                    saved = true;
                }

                if (measure_bound >= upperBound - convTol_upperBound) {
                    std::cout << "all converged at measure = " << measure_bound << ", b = " << upperBound << " lambda = " << energyParams[0] << std::endl;
                    if (iterNum_bestFeasible != iterNum) {
                        assert(iterNum_bestFeasible >= 0);
                        optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                        std::cout << "rolled back to best feasible in iter " << iterNum_bestFeasible << std::endl;
                    }
                    return false;
                }
            }
        }

        // lambda update (dual update)
        energyParams[0] = updateLambda(measure_bound, energyParams[0]);
        // TODO: needs to be careful on lambda update space

        // critical lambda scheme
        if (checkConvergence) {
            // update lambda until feasible update on T might be triggered
            if (measure_bound > upperBound) {
                // need to cut further, increase energyParams[0]
                std::cout << "curUpdated = " << energyParams[0] << ", increase" << std::endl;

                //            std::cout << "iSplit:" << std::endl;
                //            for(const auto& i : energyChanges_iSplit) {
                //                std::cout << i.first << "," << i.second << std::endl;
                //            }
                //            std::cout << "bSplit:" << std::endl;
                //            for(const auto& i : energyChanges_bSplit) {
                //                std::cout << i.first << "," << i.second << std::endl;
                //            }
                //            std::cout << "merge:" << std::endl;
                //            for(const auto& i : energyChanges_merge) {
                //                std::cout << i.first << "," << i.second << std::endl;
                //            }
                if ((!optState.energyChanges_merge.empty()) && (computeOptPicked(optState.energyChanges_bSplit, optState.energyChanges_merge, 1.0 - energyParams[0]) == 1)) {
                    // still picking merge
                    do {
                        energyParams[0] = updateLambda(measure_bound, energyParams[0]);
                    } while ((computeOptPicked(optState.energyChanges_bSplit, optState.energyChanges_merge, 1.0 - energyParams[0]) == 1));

                    std::cout << "iterativelyUpdated = " << energyParams[0] << ", increase for switch" << std::endl;
                }

                if ((!checkCand(optState.energyChanges_iSplit)) && (!checkCand(optState.energyChanges_bSplit))) {
                    // if filtering too strong
                    reQuery = true;
                    std::cout << "enlarge filtering!" << std::endl;
                } else {
                    double eDec_b, eDec_i;
                    assert(!(optState.energyChanges_bSplit.empty() && optState.energyChanges_iSplit.empty()));
                    int id_pickingBSplit = computeBestCand(optState.energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                    int id_pickingISplit = computeBestCand(optState.energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                    while ((eDec_b > 0.0) && (eDec_i > 0.0)) {
                        energyParams[0] = updateLambda(measure_bound, energyParams[0]);
                        id_pickingBSplit = computeBestCand(optState.energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                        id_pickingISplit = computeBestCand(optState.energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                    }
                    if (eDec_b <= 0.0) {
                        opType_queried = 0;
                        path_queried = optState.paths_bSplit[id_pickingBSplit];
                        newVertPos_queried = optState.newVertPoses_bSplit[id_pickingBSplit];
                    } else {
                        opType_queried = 1;
                        path_queried = optState.paths_iSplit[id_pickingISplit];
                        newVertPos_queried = optState.newVertPoses_iSplit[id_pickingISplit];
                    }

                    std::cout << "iterativelyUpdated = " << energyParams[0] << ", increased, current eDec = " << eDec_b << ", " << eDec_i << "; id: " << id_pickingBSplit << ", " << id_pickingISplit << std::endl;
                }
            } else {
                bool noOp = true;
                for (const auto ecI : optState.energyChanges_merge) {
                    if (ecI.first != __DBL_MAX__) {
                        noOp = false;
                        break;
                    }
                }
                if (noOp) {
                    std::cout << "No merge operation available, end process!" << std::endl;
                    energyParams[0] = 1.0 - eps_lambda;
                    optimizer->updateEnergyData(true, false, false);
                    if (iterNum_bestFeasible != iterNum) {
                        optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                    }
                    return false;
                }

                std::cout << "curUpdated = " << energyParams[0] << ", decrease" << std::endl;

                //!!! also account for iSplit for this switch?
                if (computeOptPicked(optState.energyChanges_bSplit, optState.energyChanges_merge, 1.0 - energyParams[0]) == 0) {
                    // still picking split
                    do {
                        energyParams[0] = updateLambda(measure_bound, energyParams[0]);
                    } while (computeOptPicked(optState.energyChanges_bSplit, optState.energyChanges_merge, 1.0 - energyParams[0]) == 0);

                    std::cout << "iterativelyUpdated = " << energyParams[0] << ", decrease for switch" << std::endl;
                }

                double eDec_m;
                assert(!optState.energyChanges_merge.empty());
                int id_pickingMerge = computeBestCand(optState.energyChanges_merge, 1.0 - energyParams[0], eDec_m);
                while (eDec_m > 0.0) {
                    energyParams[0] = updateLambda(measure_bound, energyParams[0]);
                    id_pickingMerge = computeBestCand(optState.energyChanges_merge, 1.0 - energyParams[0], eDec_m);
                }
                opType_queried = 2;
                path_queried = optState.paths_merge[id_pickingMerge];
                newVertPos_queried = optState.newVertPoses_merge[id_pickingMerge];

                std::cout << "iterativelyUpdated = " << energyParams[0] << ", decreased, current eDec = " << eDec_m << std::endl;
            }
        }

        // lambda value sanity check
        if (energyParams[0] > 1.0 - eps_lambda) {
            energyParams[0] = 1.0 - eps_lambda;
        }
        if (energyParams[0] < eps_lambda) {
            energyParams[0] = eps_lambda;
        }

        optimizer->updateEnergyData(true, false, false);

        std::cout << "measure = " << measure_bound << ", b = " << upperBound << ", updated lambda = " << energyParams[0] << std::endl;
        return true;
    }

    void postConvergeOptimization()
    {
        if (!bijectiveParam) {
            // perform exact solve
            optimizer->setAllowEDecRelTol(false);
            converged = false;
            optimizer->setPropagateFracture(false);
            while (!converged) {
                proceedOptimization(1000);
            }
        }
    }

    bool optimizationStep()
    {
        while (!converged) {
            proceedOptimization();
        }

        double stretch_l2, stretch_inf, stretch_shear, compress_inf;
        triSoup[channel_result]->computeStandardStretch(stretch_l2, stretch_inf, stretch_shear, compress_inf);
        double measure_bound = optimizer->getLastEnergyVal(true) / energyParams[0];

        switch (optState.methodType) {
        case OptCuts::MT_EBCUTS: {
            if (measure_bound <= upperBound) {
                std::cout << "measure reaches user specified upperbound " << upperBound << std::endl;

                infoName = "finalResult";
                // perform exact solve
                optimizer->setAllowEDecRelTol(false);
                converged = false;
                while (!converged) {
                    proceedOptimization(1000);
                }
            } else {
                infoName = std::to_string(iterNum);

                // continue to make geometry image cuts
                bool returnVal = optimizer->createFracture(fracThres, false, false);
                assert(returnVal);
                converged = false;
            }
            optimizer->flushEnergyFileOutput();
            optimizer->flushGradFileOutput();
            break;
        }

        case OptCuts::MT_OPTCUTS_NODUAL:
        case OptCuts::MT_OPTCUTS: {
            infoName = std::to_string(iterNum);
            if (converged == 2) {
                converged = 0;
                return false;
            }

            if ((optState.methodType == OptCuts::MT_OPTCUTS) && (measure_bound <= upperBound)) {
                // save info once bound is reached for comparison
                static bool saved = false;
                if (!saved) {
                    saved = true;
                }
            }

            // if necessary, turn on scaffolding for random one point initial cut
            if (!optimizer->isScaffolding() && bijectiveParam && rand1PInitCut) {
                optimizer->setScaffolding(true);
            }

            double E_se;
            triSoup[channel_result]->computeSeamSparsity(E_se);
            E_se /= triSoup[channel_result]->virtualRadius;
            const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];

            std::cout << iterNum << ": " << E_SD << " " << E_se << " " << triSoup[channel_result]->V_rest.rows() << std::endl;
            std::cout << iterNum << ": " << E_SD << " " << E_se << " " << triSoup[channel_result]->V_rest.rows() << std::endl;
            optimizer->flushEnergyFileOutput();
            optimizer->flushGradFileOutput();

            // continue to split boundary
            if ((optState.methodType == OptCuts::MT_OPTCUTS) && (!updateLambda_stationaryV())) {
                // oscillation detected
                postConvergeOptimization();
            } else {
                std::cout << "boundary op V " << triSoup[channel_result]->V_rest.rows() << std::endl;
                if (optimizer->createFracture(fracThres, false, topoLineSearch)) {
                    converged = false;
                } else {
                    // if no boundary op, try interior split if split is the current best boundary op
                    if ((measure_bound > upperBound) && optimizer->createFracture(fracThres, false, topoLineSearch, true)) {
                        std::cout << "interior split " << triSoup[channel_result]->V_rest.rows() << std::endl;
                        converged = false;
                    } else {
                        if ((optState.methodType == OptCuts::MT_OPTCUTS_NODUAL) || (!updateLambda_stationaryV(false, true))) {
                            // all converged
                            postConvergeOptimization();
                        } else {
                            // split or merge after lambda update
                            if (reQuery) {
                                optState.filterExp_in += std::log(2.0) / std::log(optState.inSplitTotalAmt);
                                optState.filterExp_in = std::min(1.0, optState.filterExp_in);
                                while (!optimizer->createFracture(fracThres, false, topoLineSearch, true)) {
                                    optState.filterExp_in += std::log(2.0) / std::log(optState.inSplitTotalAmt);
                                    optState.filterExp_in = std::min(1.0, optState.filterExp_in);
                                }
                                reQuery = false;
                                // TODO: set filtering param back?
                            } else {
                                optimizer->createFracture(opType_queried, path_queried, newVertPos_queried, topoLineSearch);
                            }
                            opType_queried = -1;
                            converged = false;
                        }
                    }
                }
            }
            break;
        }

        case OptCuts::MT_DISTMIN: {
            postConvergeOptimization();
            break;
        }
        }
        return converged;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> optimize_mesh(Eigen::MatrixXd& vertices, Eigen::MatrixXd& uvs, Eigen::MatrixXi& indices, Eigen::MatrixXi& uv_indices)
    {
        Eigen::MatrixXd V, UV;
        Eigen::MatrixXi F, FUV;

        V = vertices;
        UV = uvs;
        F = indices;
        FUV = uv_indices;

        Eigen::VectorXi B;
        bool isManifold = igl::is_vertex_manifold(F, B) && igl::is_edge_manifold(F);
        if (!isManifold) {
            std::cout << "input mesh contains non-manifold edges or vertices" << std::endl;
            std::cout << "please cleanup the mesh and retry" << std::endl;
            throw std::runtime_error("non-manifold mesh");
        }

        // Argument 3: lambda
        lambda_init = 0.999;

        // Argument 4: testID
        double testID = 1.0; // test id for naming result folder

        // Argument 5: optState.methodType
        optState.methodType = OptCuts::MethodType::MT_OPTCUTS;

        std::string startDS;
        switch (optState.methodType) {
        case OptCuts::MT_OPTCUTS_NODUAL:
            startDS = "OptCuts_noDual";
            break;

        case OptCuts::MT_OPTCUTS:
            startDS = "OptCuts";
            break;

        case OptCuts::MT_EBCUTS:
            startDS = "EBCuts";
            bijectiveParam = false;
            break;

        case OptCuts::MT_DISTMIN:
            lambda_init = 0.0;
            startDS = "DistMin";
            break;

        default:
            throw std::runtime_error("invalid method type");
            break;
        }

        // Argument 6: upperBound
        upperBound = 4.1;

        // Argument 7: bijectiveParam
        bijectiveParam = 1;

        // Argument 8: initCutOption
        initCutOption = 0;
        switch (initCutOption) {
        case 0:
            std::cout << "random 2-edge initial cut for genus-0 closed surface" << std::endl;
            break;

        case 1:
            std::cout << "farthest 2-point initial cut for genus-0 closed surface" << std::endl;
            break;

        default:
            std::cout << "input initial cut option invalid, use default" << std::endl;
            std::cout << "random 2-edge initial cut for genus-0 closed surface" << std::endl;
            initCutOption = 0;
            break;
        }

        std::string folderTail = "";

        //////////////////////////////////
        // initialize UV

        if (UV.rows() != 0) {
            // with input UV
            OptCuts::TriMesh* temp = new OptCuts::TriMesh(optState, V, F, UV, FUV, false);

            std::vector<std::vector<int>> bnd_all;
            igl::boundary_loop(temp->F, bnd_all);

            // TODO: check input UV genus (validity)
            // right now OptCuts assumes input UV is a set of topological disks

            bool recompute_UV_needed = !temp->checkInversion();
            if ((!recompute_UV_needed) && bijectiveParam && (bnd_all.size() > 1)) {
                // TODO: check overlaps and decide whether needs recompute UV
                // needs to check even if bnd_all.size() == 1
                // right now OptCuts take the input seams and recompute UV by default when bijective mapping is enabled
                recompute_UV_needed = true;
            }
            if (recompute_UV_needed) {
                std::cout << "local injectivity violated in given input UV map, " << "or multi-chart bijective UV map needs to be ensured, " << "obtaining new initial UV map by applying Tutte's embedding..." << std::endl;

                int UVGridDim = 0;
                do {
                    ++UVGridDim;
                } while (UVGridDim * UVGridDim < bnd_all.size());
                std::cout << "UVGridDim " << UVGridDim << std::endl;

                Eigen::VectorXi bnd_stacked;
                Eigen::MatrixXd bnd_uv_stacked;
                for (int bndI = 0; bndI < bnd_all.size(); bndI++) {
                    // map boundary to unit circle
                    bnd_stacked.conservativeResize(bnd_stacked.size() + bnd_all[bndI].size());
                    bnd_stacked.tail(bnd_all[bndI].size()) = Eigen::VectorXi::Map(bnd_all[bndI].data(),
                        bnd_all[bndI].size());

                    Eigen::MatrixXd bnd_uv;
                    igl::map_vertices_to_circle(temp->V_rest,
                        bnd_stacked.tail(bnd_all[bndI].size()),
                        bnd_uv);
                    double xOffset = bndI % UVGridDim * 2.1, yOffset = bndI / UVGridDim * 2.1;
                    for (int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                        bnd_uv(bnd_uvI, 0) += xOffset;
                        bnd_uv(bnd_uvI, 1) += yOffset;
                    }
                    bnd_uv_stacked.conservativeResize(bnd_uv_stacked.rows() + bnd_uv.rows(), 2);
                    bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;
                }

                // Harmonic map with uniform weights
                Eigen::SparseMatrix<double> A, M;
                OptCuts::IglUtils::computeUniformLaplacian(temp->F, A);
                igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, temp->V);

                if (!temp->checkInversion()) {
                    throw std::runtime_error("local injectivity still violated in the computed initial UV map, please carefully check UV topology for e.g. non-manifold vertices.");
                }
            }

            triSoup.emplace_back(temp);
        } else {
            // no UV provided, compute initial UV

            Eigen::VectorXi C;
            igl::facet_components(F, C);
            int n_components = C.maxCoeff() + 1;
            std::cout << n_components << " disconnected components in total" << std::endl;

            // in each pass, make one cut on each component if needed, until all becoming disk-topology
            OptCuts::TriMesh temp(optState, V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
            std::vector<Eigen::MatrixXi> F_component(n_components);
            std::vector<std::set<int>> V_ind_component(n_components);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                F_component[C[triI]].conservativeResize(F_component[C[triI]].rows() + 1, 3);
                F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
                for (int i = 0; i < 3; ++i) {
                    V_ind_component[C[triI]].insert(temp.F(triI, i));
                }
            }
            while (true) {
                std::vector<int> components_to_cut;
                for (int componentI = 0; componentI < n_components; ++componentI) {
                    std::cout << ">>> component " << componentI << std::endl;

                    int EC = igl::euler_characteristic(temp.V, F_component[componentI]) - temp.V.rows() + V_ind_component[componentI].size();
                    std::cout << "euler_characteristic " << EC << std::endl;
                    if (EC < 1) {
                        // treat as higher-genus surfaces using cut_to_disk()
                        components_to_cut.emplace_back(-componentI - 1);
                    } else if (EC == 2) {
                        // closed genus-0 surface
                        components_to_cut.emplace_back(componentI);
                    } else if (EC != 1) {
                        throw std::runtime_error("unsupported single-connected component");
                    }
                }
                std::cout << components_to_cut.size() << " components to cut to disk" << std::endl;

                if (components_to_cut.empty()) {
                    break;
                }

                for (auto componentI : components_to_cut) {
                    if (componentI < 0) {
                        // cut high genus
                        componentI = -componentI - 1;

                        std::vector<std::vector<int>> cuts;
                        igl::cut_to_disk(F_component[componentI], cuts); // Meshes with boundary are supported; boundary edges will be included as cuts.
                        std::cout << cuts.size() << " seams to cut component " << componentI << std::endl;

                        // only cut one seam each time to avoid seam vertex id inconsistency
                        int cuts_made = 0;
                        for (auto& seamI : cuts) {
                            if (seamI.front() == seamI.back()) {
                                // cutPath() dos not support closed-loop cuts, split it into two cuts
                                cuts_made += temp.cutPath(std::vector<int>({ seamI[seamI.size() - 3], seamI[seamI.size() - 2], seamI[seamI.size() - 1] }), true);
                                temp.initSeams = temp.cohE;
                                seamI.resize(seamI.size() - 2);
                            }
                            cuts_made += temp.cutPath(seamI, true);
                            temp.initSeams = temp.cohE;
                            if (cuts_made) {
                                break;
                            }
                        }

                        if (!cuts_made) {
                            throw std::runtime_error("no cuts made when cutting input geometry to disk-topology");
                        }
                    } else {
                        // cut the topological sphere into a topological disk
                        switch (initCutOption) {
                        case 0:
                            temp.onePointCut(F_component[componentI](0, 0));
                            rand1PInitCut = (n_components == 1);
                            break;

                        case 1:
                            temp.farthestPointCut(F_component[componentI](0, 0));
                            break;

                        default:
                            std::cout << "invalid initCutOption " << initCutOption << std::endl;
                            assert(0);
                            break;
                        }
                    }
                }

                // data update on each component for identifying a new cut
                F_component.resize(0);
                F_component.resize(n_components);
                V_ind_component.resize(0);
                V_ind_component.resize(n_components);
                for (int triI = 0; triI < temp.F.rows(); ++triI) {
                    F_component[C[triI]].conservativeResize(F_component[C[triI]].rows() + 1, 3);
                    F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
                    for (int i = 0; i < 3; ++i) {
                        V_ind_component[C[triI]].insert(temp.F(triI, i));
                    }
                }
            }

            int UVGridDim = 0;
            do {
                ++UVGridDim;
            } while (UVGridDim * UVGridDim < n_components);
            std::cout << "UVGridDim " << UVGridDim << std::endl;

            // compute boundary UV coordinates, using a grid layout for muliComp
            Eigen::VectorXi bnd_stacked;
            Eigen::MatrixXd bnd_uv_stacked;
            for (int componentI = 0; componentI < n_components; ++componentI) {
                std::cout << ">>> component " << componentI << std::endl;

                std::vector<std::vector<int>> bnd_all;
                igl::boundary_loop(F_component[componentI], bnd_all);
                std::cout << "boundary loop count " << bnd_all.size() << std::endl; // must be 1 for the current initial cut strategy

                int longest_bnd_id = 0;
                for (int bnd_id = 1; bnd_id < bnd_all.size(); ++bnd_id) {
                    if (bnd_all[longest_bnd_id].size() < bnd_all[bnd_id].size()) {
                        longest_bnd_id = bnd_id;
                    }
                }
                std::cout << "longest_bnd_id " << longest_bnd_id << std::endl;

                bnd_stacked.conservativeResize(bnd_stacked.size() + bnd_all[longest_bnd_id].size());
                bnd_stacked.tail(bnd_all[longest_bnd_id].size()) = Eigen::VectorXi::Map(
                    bnd_all[longest_bnd_id].data(), bnd_all[longest_bnd_id].size());

                Eigen::MatrixXd bnd_uv;
                igl::map_vertices_to_circle(temp.V_rest,
                    bnd_stacked.tail(bnd_all[longest_bnd_id].size()),
                    bnd_uv);
                double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;
                for (int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                    bnd_uv(bnd_uvI, 0) += xOffset;
                    bnd_uv(bnd_uvI, 1) += yOffset;
                }
                bnd_uv_stacked.conservativeResize(bnd_uv_stacked.rows() + bnd_uv.rows(), 2);
                bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;
            }

            // Harmonic map with uniform weights
            Eigen::MatrixXd UV_Tutte;
            Eigen::SparseMatrix<double> A, M;
            OptCuts::IglUtils::computeUniformLaplacian(temp.F, A);
            igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, UV_Tutte);

            triSoup.emplace_back(new OptCuts::TriMesh(optState, V, F, UV_Tutte, temp.F, false));
        }

        // initialize UV
        //////////////////////////////////

        mkdir(outputFolderPath.c_str(), 0777);
        outputFolderPath += '/';
        igl::writeOBJ(outputFolderPath + "initial_cuts.obj", triSoup.back()->V_rest, triSoup.back()->F);

        // setup timer
        optState.timer.new_activity("topology");
        optState.timer.new_activity("descent");
        optState.timer.new_activity("scaffolding");
        optState.timer.new_activity("energyUpdate");

        optState.timer_step.new_activity("matrixComputation");
        optState.timer_step.new_activity("matrixAssembly");
        optState.timer_step.new_activity("symbolicFactorization");
        optState.timer_step.new_activity("numericalFactorization");
        optState.timer_step.new_activity("backSolve");
        optState.timer_step.new_activity("lineSearch");
        optState.timer_step.new_activity("boundarySplit");
        optState.timer_step.new_activity("interiorSplit");
        optState.timer_step.new_activity("cornerMerge");

        // * Our approach
        texScale = 10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
        energyParams.emplace_back(1.0 - lambda_init);
        energyTerms.emplace_back(new OptCuts::SymDirichletEnergy());

        optimizer = new OptCuts::Optimizer(optState, *triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself

        optimizer->precompute();
        triSoup.emplace_back(&optimizer->getResult());
        triSoup_backup = optimizer->getResult();
        triSoup.emplace_back(&optimizer->getData_findExtrema()); // for visualizing UV map for finding extrema
        if (lambda_init > 0.0) {
            // fracture mode
            optState.fractureMode = true;
        }

        while (!optimizationStep()) {
        }

        // Get the result
        const auto meshData = triSoup[channel_result]->getMeshData(F, true);

        for (auto& eI : energyTerms) {
            delete eI;
        }
        delete optimizer;
        delete triSoup[0];
        return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> { meshData.V, meshData.UV, meshData.F, meshData.FUV };
    }

}; // struct OptCuts
}
