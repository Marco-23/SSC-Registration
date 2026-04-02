/*
===============================================================================
SSC fragment quality evaluation
-------------------------------------------------------------------------------
This file computes the numerical quality measures used for the SSC acceptance
decision.

Reference article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
This file converts the output of the semi-sphere seed clustering into practical
acceptance metrics, such as:

- mean seed displacement,
- standard deviation of displacement,
- optional weighted variants,
- fragment quality labels or threshold-based decisions.


===============================================================================
*/


#include "calculate_quality.h"
#include "utilities.h"

#include <iostream>

SphereQualityResult calculate_sphere_quality(
    const std::vector<double>& total_displacement,
    const std::vector<int>& cluster_counts,
    int min_populated_seeds,
    int min_seed_points,
    double max_mean_displacement,
    double max_std_displacement)
{
    SphereQualityResult result;
    const auto stats = computeDisplacementStats(total_displacement);
    result.mean_displacement = stats.mean_displacement;
    result.std_displacement = stats.std_displacement;

    for (int cluster_size : cluster_counts) {
        if (cluster_size >= min_seed_points) result.populated_seeds++;
    }

    result.accepted = (result.populated_seeds >= min_populated_seeds) &&
                      (result.mean_displacement <= max_mean_displacement) &&
                      (result.std_displacement <= max_std_displacement);
    result.label = result.accepted ? "Good" : "Bad";

    std::cout << "  SSC quality summary"
              << " | populated_seeds=" << result.populated_seeds
              << " (required>=" << min_populated_seeds << ")"
              << " | mean_disp=" << result.mean_displacement
              << " (limit=" << max_mean_displacement << ")"
              << " | std_disp=" << result.std_displacement
              << " (limit=" << max_std_displacement << ")"
              << " | decision=" << result.label << "\n";
    return result;
}
