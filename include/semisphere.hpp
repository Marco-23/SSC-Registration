#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

struct SSCOptions {
    double frame_sec             = 10.0;
    double min_traj_len_m        = 10.0;
    int    max_extra_frames      = 10;
    int    normal_knn            = 20;
    double seed_tol              = 1e-3;
    int    kmeans_max_iter       = 50;
    double seed_stddev_thresh    = 0.25;
    int    min_populated_seeds   = 3;
    std::size_t max_points_eval  = 300000;

    std::size_t min_points_for_eval = 5000;
    double      accept_if_extent_over_m = 200.0;

    // NEW: a seed only “counts” if it has at least this many normals
    std::size_t min_points_per_seed = 20000;

    bool   enforce_forward_time_increasing_x = true;
    bool   bake_alignment_into_xyz           = false;
};

struct SSCFragment {
    int start_frame = 0;
    int end_frame   = 0;

    double tmin     = 0.0;
    double tmax     = 0.0;
    bool   accepted = false;

    // Diagnostics
    int    populated_seeds       = 0;   // NOW = seeds with count >= min_points_per_seed
    double seed_disp_mean        = 0.0;
    double seed_disp_std         = 0.0;
    double principal_extent_m    = 0.0;
    Eigen::Matrix3d R_align      = Eigen::Matrix3d::Identity();
    Eigen::Vector3d c_align      = Eigen::Vector3d::Zero();

    std::vector<int> seed_counts;       // raw counts [+X,-X,+Y,-Y,+Z]
    int normals_used = 0;

    std::vector<Eigen::Vector3d> final_seeds;

    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz;  // unaligned unless bake_alignment_into_xyz=true
};

bool run_semi_sphere_check(const pcl::PCLPointCloud2& source_with_gps,
                           const SSCOptions& opt,
                           std::vector<SSCFragment>& out);

void print_ssc_summary(const std::vector<SSCFragment>& frags, bool onlyAccepted);
void print_ssc_summary(const std::vector<SSCFragment>& frags);


