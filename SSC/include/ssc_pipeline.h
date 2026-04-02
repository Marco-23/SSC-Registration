/*
===============================================================================
SSC pipeline interface
-------------------------------------------------------------------------------
This header defines the public data structures and function interfaces for the
SSC-based adaptive fragmentation workflow.

Reference article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
This file exposes the configurable parameters and result structures used by the
SSC pipeline. These parameters control:

- initial time splitting,
- normal estimation,
- seed-population requirements,
- SSC quality thresholds,
- fragment growth limits,
- project-specific acceptance overrides.

===============================================================================
*/

#pragma once
#include "calculate_quality.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

// Main user-facing SSC parameters.
// These defaults define the behavior of the standalone pipeline.
// They can all be overridden from the command line.
struct SSCParams {
    double split_seconds = 10.0;
    double max_range = 20.0;
    bool use_range_filter = true;
    int normal_k = 300;
    int kmeans_max_iterations = 100;
    double kmeans_tolerance = 0.17;
    double min_fragment_extent_m = 10.0;
    double accept_if_extent_over_m = 200.0;
    int max_extra_frames = 100;
    int min_populated_seeds = 4;
    int min_seed_points = 1000;
    double max_mean_displacement = 0.24;
    double max_std_displacement = 0.12;
    bool save_initial_frames = true;
    bool save_cluster_csv = true;
    bool save_preview_ply = true;
    bool show_viewer = false;
};

// One initial GPS-split frame before adaptive SSC merging.
struct InitialFrame {
    pcl::PCLPointCloud2 blob;
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz;
    double t0 = 0.0;
    double t1 = 0.0;
    int index = -1;
    std::string name;
};

// Result for one final fragment candidate after SSC evaluation.
// This stores both the geometry and the diagnostic values used for accept/reject.
struct SSCFragmentResult {
    pcl::PCLPointCloud2 blob;
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz;
    int start_initial_frame = -1;
    int end_initial_frame = -1;
    double gps_t0 = 0.0;
    double gps_t1 = 0.0;
    double extent_m = 0.0;
    SphereQualityResult quality;
    std::vector<int> cluster_assignments;
    std::vector<int> cluster_counts;
    std::vector<double> seed_displacements;
    bool accepted = false;
};

bool buildInitialFramesFromGPS(
    const pcl::PCLPointCloud2& sorted_blob,
    double split_seconds,
    std::vector<InitialFrame>& out_frames,
    std::string& err);

double computeFragmentExtentMeters(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud);

bool runSSCOnFragment(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input_cloud,
    const SSCParams& params,
    SSCFragmentResult& out,
    std::string& err);

void saveFragmentSummaryCsv(const std::string& path, const std::vector<SSCFragmentResult>& results);
