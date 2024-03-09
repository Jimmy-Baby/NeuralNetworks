#pragma once

#include <eigen3/Eigen/Core>

#ifdef USE_32_BIT_PRECISION
using TScalar = float;
#else
using TScalar = double;
#endif

using TMatrix = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;
using TVector = Eigen::Matrix<TScalar, Eigen::Dynamic, 1>;
using TIntegerVector = Eigen::Matrix<int, 1, Eigen::Dynamic>;
