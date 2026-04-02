/*
===============================================================================
SSC semi-sphere seed clustering interface
-------------------------------------------------------------------------------
This header declares the clustering routine used by the SSC evaluation.

Reference article:
  Ortiz Rincón, Yang, Holst (2026)
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
The declared function takes normal directions on the SSC semi-sphere and
returns:

- cluster assignments,
- updated seed positions,
- seed displacement values,
- per-seed population counts.

These outputs are later used to assess whether a candidate fragment should be
accepted or extended.
===============================================================================
*/


#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <vector>

std::tuple<std::vector<int>, std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<int>>
dynamic_seed_growth(
    const std::vector<Eigen::Vector3d>& normals,
    const std::vector<Eigen::Vector3d>& seed_points,
    int max_iterations = 100,
    double tolerance = 1e-4);
