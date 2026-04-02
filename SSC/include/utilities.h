/*
===============================================================================
SSC math utilities interface
-------------------------------------------------------------------------------
This header declares small helper functions used throughout the SSC codebase.

Reference article:
  Ortiz Rincón, Yang, Holst (2026)
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
These declarations provide shared access to small numerical helper routines used
by the SSC pipeline.
===============================================================================
*/


#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

Eigen::Vector3d safeNormalize(const Eigen::Vector3d& vec);
double distance3(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
std::string toLowerCopy(std::string s);

struct SeedDispersionStats {
    double mean_displacement = 0.0;
    double std_displacement = 0.0;
};

SeedDispersionStats computeDisplacementStats(const std::vector<double>& displacements);
