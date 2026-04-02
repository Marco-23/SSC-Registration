/*
===============================================================================
SSC quality evaluation interface
-------------------------------------------------------------------------------
This header declares the quality-measure functions used to decide whether a
candidate fragment passes the SSC test.

Reference article:
  Ortiz Rincón, Yang, Holst (2026)
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
These functions translate clustering outputs into fragment-quality metrics that
can be compared against user-defined thresholds.
===============================================================================
*/

#pragma once
#include <string>
#include <vector>

struct SphereQualityResult {
    bool accepted = false;
    double mean_displacement = 0.0;
    double std_displacement = 0.0;
    int populated_seeds = 0;
    std::string label;
};

SphereQualityResult calculate_sphere_quality(
    const std::vector<double>& total_displacement,
    const std::vector<int>& cluster_counts,
    int min_populated_seeds,
    int min_seed_points,
    double max_mean_displacement,
    double max_std_displacement);
