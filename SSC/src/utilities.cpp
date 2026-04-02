/*
===============================================================================
SSC math utilities
-------------------------------------------------------------------------------
This file provides small mathematical helper functions used by the SSC
implementation.

Reference article:
  Ortiz Rincón, Yang, Holst (2026)
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
This file contains low-level helper operations such as vector normalization and
distance computations that are used repeatedly in the semi-sphere clustering
and quality evaluation steps.

===============================================================================
*/


#include "utilities.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>

Eigen::Vector3d safeNormalize(const Eigen::Vector3d& vec) {
    const double n = vec.norm();
    if (n <= 1e-12) return Eigen::Vector3d::Zero();
    return vec / n;
}

double distance3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).norm();
}

std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

SeedDispersionStats computeDisplacementStats(const std::vector<double>& displacements) {
    SeedDispersionStats out;
    if (displacements.empty()) return out;

    const double sum = std::accumulate(displacements.begin(), displacements.end(), 0.0);
    out.mean_displacement = sum / static_cast<double>(displacements.size());

    double var = 0.0;
    for (double v : displacements) {
        const double d = v - out.mean_displacement;
        var += d * d;
    }
    out.std_displacement = std::sqrt(var / static_cast<double>(displacements.size()));
    return out;
}
