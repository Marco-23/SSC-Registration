/*
===============================================================================
SSC semi-sphere seed clustering
-------------------------------------------------------------------------------
This file implements the iterative five-seed clustering step used in the SSC
evaluation.

Reference article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
Given a set of normal directions projected to the SSC semi-sphere, this file:

- initializes five canonical seed directions,
- assigns each normal to the nearest seed,
- updates the seed positions iteratively,
- measures seed displacement from the original canonical directions,
- reports cluster populations and displacement values.

Why this matters
----------------
The goal of SSC is to verify that a candidate fragment contains sufficiently
distributed orthogonal geometric structure. This clustering step is the core
test that checks whether the fragment contains a meaningful spread of normal
directions.

Canonical seed directions
-------------------------
The five seed directions are:

  (+x), (-x), (+y), (-y), (+z)

These correspond to the expected dominant directions in many urban MLS scenes,
such as opposing facades and horizontal ground or roof surfaces.

===============================================================================
*/



#include "dynamic_seed_growth.h"
#include "utilities.h"

#include <algorithm>
#include <limits>
#include <vector>

std::tuple<std::vector<int>, std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<int>> dynamic_seed_growth(
    const std::vector<Eigen::Vector3d>& normals,
    const std::vector<Eigen::Vector3d>& seed_points,
    int max_iterations,
    double tolerance)
{
    const int num_seeds = static_cast<int>(seed_points.size());
    const int num_normals = static_cast<int>(normals.size());

    std::vector<Eigen::Vector3d> original_seed_points;
    original_seed_points.reserve(seed_points.size());
    std::vector<Eigen::Vector3d> current_seed_points = seed_points;
    for (const auto& seed : seed_points) {
        original_seed_points.push_back(safeNormalize(seed));
    }

    std::vector<Eigen::Vector3d> normalized_normals;
    normalized_normals.reserve(normals.size());
    for (const auto& normal : normals) {
        normalized_normals.push_back(safeNormalize(normal));
    }

    std::vector<int> cluster_assignments(num_normals, -1);
    std::vector<int> cluster_counts(num_seeds, 0);
    std::vector<Eigen::Vector3d> new_seed_points(num_seeds, Eigen::Vector3d::Zero());

    const double tolerance_sq = tolerance * tolerance;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        for (int i = 0; i < num_normals; ++i) {
            double min_dist_sq = std::numeric_limits<double>::infinity();
            int closest_seed = -1;
            for (int j = 0; j < num_seeds; ++j) {
                const Eigen::Vector3d d = current_seed_points[j] - normalized_normals[i];
                const double dist_sq = d.squaredNorm();
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_seed = j;
                }
            }
            cluster_assignments[i] = closest_seed;
        }

        std::fill(new_seed_points.begin(), new_seed_points.end(), Eigen::Vector3d::Zero());
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

        for (int i = 0; i < num_normals; ++i) {
            const int cluster_id = cluster_assignments[i];
            if (cluster_id < 0) continue;
            new_seed_points[cluster_id] += normalized_normals[i];
            cluster_counts[cluster_id]++;
        }

        for (int i = 0; i < num_seeds; ++i) {
            if (cluster_counts[i] > 0) {
                new_seed_points[i] /= static_cast<double>(cluster_counts[i]);
                new_seed_points[i] = safeNormalize(new_seed_points[i]);
            } else {
                new_seed_points[i] = current_seed_points[i];
            }
        }

        double max_displacement_sq = 0.0;
        for (int i = 0; i < num_seeds; ++i) {
            const double displacement_sq = (new_seed_points[i] - current_seed_points[i]).squaredNorm();
            max_displacement_sq = std::max(max_displacement_sq, displacement_sq);
        }
        current_seed_points = new_seed_points;
        if (max_displacement_sq < tolerance_sq) break;
    }

    std::vector<double> total_displacement(num_seeds, 0.0);
    for (int i = 0; i < num_seeds; ++i) {
        total_displacement[i] = distance3(current_seed_points[i], original_seed_points[i]);
    }

    return {cluster_assignments, current_seed_points, total_displacement, cluster_counts};
}
