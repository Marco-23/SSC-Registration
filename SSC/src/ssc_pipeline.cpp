/*
===============================================================================
SSC pipeline core
-------------------------------------------------------------------------------
This file implements the main geometric logic of the SSC-based adaptive
fragmentation workflow for MLS point clouds.

Reference article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
This file contains the core processing steps used to evaluate whether a
candidate fragment should be accepted or extended:

- compute fragment extent,
- align the fragment according to its dominant horizontal direction,
- estimate normals on the whole current candidate fragment,
- orient normals consistently,
- project the normal directions onto the SSC semi-sphere,
- evaluate the directional distribution using the five-seed clustering logic,
- return the metrics used for the acceptance decision.

Important note
--------------
This implementation follows the SSC idea presented in the article, but may
include project-specific engineering extensions for practical workflow use.
Examples include configurable thresholds and optional automatic acceptance when
the fragment extent exceeds a user-defined maximum value.

Conceptual role in the workflow
-------------------------------
The overall pipeline is:

  load cloud
    -> detect GPS time
    -> split into initial time frames
    -> build candidate fragment
    -> call SSC evaluation from this file
    -> accept fragment or append next frame
    -> save accepted fragment

This file focuses only on the SSC evaluation part of that process.
===============================================================================
*/




#include "ssc_pipeline.h"
#include "dynamic_seed_growth.h"
#include <algorithm>
#include "file_io.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#if defined(__has_include)
#  if __has_include(<pcl/features/normal_3d_omp.h>)
#    include <pcl/features/normal_3d_omp.h>
#    define SSC_HAS_NORMAL_OMP 1
#  else
#    define SSC_HAS_NORMAL_OMP 0
#  endif
#else
#  define SSC_HAS_NORMAL_OMP 0
#endif
#include <pcl/search/kdtree.h>

bool buildInitialFramesFromGPS(
    const pcl::PCLPointCloud2& sorted_blob,
    double split_seconds,
    std::vector<InitialFrame>& out_frames,
    std::string& err)
{
    std::vector<double> gps;
    if (!readFieldAsDoubleVector(sorted_blob, "gps_time", gps) && !readFieldAsDoubleVector(sorted_blob, "Gps_Time", gps)) {
        err = "Could not find gps_time/Gps_Time field for GPS-based splitting.";
        return false;
    }
    if (gps.empty()) { err = "GPS time field is empty."; return false; }

    std::vector<int> current;
    double start_t = gps.front();
    int frame_index = 0;

    auto flush_frame = [&](double t0, double t1) {
        if (current.empty()) return;
        InitialFrame f;
        f.index = frame_index;
        f.t0 = t0;
        f.t1 = t1;
        f.name = "frame_" + std::to_string(frame_index);
        f.blob = sliceCloud2ByIndices(sorted_blob, current);
        f.xyz.reset(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromPCLPointCloud2(f.blob, *f.xyz);
        out_frames.push_back(std::move(f));
        ++frame_index;
        current.clear();
    };

    for (size_t i = 0; i < gps.size(); ++i) {
        const double t = gps[i];
        if (current.empty()) {
            start_t = t;
            current.push_back(static_cast<int>(i));
            continue;
        }
        if (t - start_t <= split_seconds) current.push_back(static_cast<int>(i));
        else {
            flush_frame(start_t, gps[i - 1]);
            start_t = t;
            current.push_back(static_cast<int>(i));
        }
    }
    if (!current.empty()) flush_frame(start_t, gps.back());
    if (out_frames.empty()) { err = "No initial frames were created from GPS splitting."; return false; }
    return true;
}

double computeFragmentExtentMeters(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    if (!cloud || cloud->empty()) return 0.0;
    Eigen::Vector2d mean(0.0, 0.0);
    for (const auto& p : cloud->points) { mean.x() += p.x; mean.y() += p.y; }
    mean /= static_cast<double>(cloud->size());

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& p : cloud->points) {
        const Eigen::Vector2d d(p.x - mean.x(), p.y - mean.y());
        cov += d * d.transpose();
    }
    cov /= std::max<double>(1.0, static_cast<double>(cloud->size() - 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
    Eigen::Vector2d axis = es.eigenvectors().col(1).normalized();

    double minp = std::numeric_limits<double>::infinity();
    double maxp = -std::numeric_limits<double>::infinity();
    for (const auto& p : cloud->points) {
        const double proj = axis.dot(Eigen::Vector2d(p.x, p.y));
        minp = std::min(minp, proj);
        maxp = std::max(maxp, proj);
    }
    return std::max(0.0, maxp - minp);
}

static Eigen::Matrix4f alignDominantXYAxisToX(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    Eigen::Vector2d mean(0.0, 0.0);
    for (const auto& p : cloud->points) { mean.x() += p.x; mean.y() += p.y; }
    mean /= static_cast<double>(cloud->size());

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& p : cloud->points) {
        const Eigen::Vector2d d(p.x - mean.x(), p.y - mean.y());
        cov += d * d.transpose();
    }
    cov /= std::max<double>(1.0, static_cast<double>(cloud->size() - 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
    Eigen::Vector2d axis = es.eigenvectors().col(1).normalized();
    const double yaw = std::atan2(axis.y(), axis.x());
    const float c = static_cast<float>(std::cos(-yaw));
    const float s = static_cast<float>(std::sin(-yaw));

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(0,0) = c; T(0,1) = -s;
    T(1,0) = s; T(1,1) =  c;

    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
    T1(0,3) = static_cast<float>(-mean.x());
    T1(1,3) = static_cast<float>(-mean.y());
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2(0,3) = static_cast<float>(mean.x());
    T2(1,3) = static_cast<float>(mean.y());
    return T2 * T * T1;
}

static void orientNormalsTowardCentroid(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    Eigen::Vector3d centroid(0.0, 0.0, 0.0);
    for (const auto& p : cloud->points) centroid += Eigen::Vector3d(p.x, p.y, p.z);
    centroid /= static_cast<double>(cloud->size());

    for (size_t i = 0; i < normals->size() && i < cloud->size(); ++i) {
        auto& n = normals->points[i];
        const auto& p = cloud->points[i];
        const Eigen::Vector3d v_to_centroid = centroid - Eigen::Vector3d(p.x, p.y, p.z);
        const Eigen::Vector3d nv(n.normal_x, n.normal_y, n.normal_z);
        if (nv.dot(v_to_centroid) < 0.0) {
            n.normal_x *= -1.0f; n.normal_y *= -1.0f; n.normal_z *= -1.0f;
        }
    }
}

bool runSSCOnFragment(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input_cloud, const SSCParams& params, SSCFragmentResult& out, std::string& err)
{
    if (!input_cloud || input_cloud->empty()) { err = "runSSCOnFragment: input cloud is empty."; return false; }
    out.xyz.reset(new pcl::PointCloud<pcl::PointXYZ>(*input_cloud));
    out.extent_m = computeFragmentExtentMeters(input_cloud);

    const Eigen::Matrix4f T = alignDominantXYAxisToX(input_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*input_cloud, *aligned, T);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
#if SSC_HAS_NORMAL_OMP
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
#else
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
#endif
    ne.setInputCloud(aligned);
    ne.setSearchMethod(tree);
    ne.setKSearch(std::max(5, params.normal_k));
    ne.compute(*normals);
    if (normals->size() != aligned->size()) { err = "Failed to compute normals for all points."; return false; }
    orientNormalsTowardCentroid(aligned, normals);

    std::vector<Eigen::Vector3d> eigen_normals;
    eigen_normals.reserve(normals->size());
    for (size_t i = 0; i < normals->size(); ++i) {
        const auto& n = normals->points[i];
        eigen_normals.emplace_back(n.normal_x, n.normal_y, std::abs(n.normal_z));
    }
    const std::vector<Eigen::Vector3d> seed_points = {
        Eigen::Vector3d( 1, 0, 0), Eigen::Vector3d( 0, 1, 0), Eigen::Vector3d( 0, 0, 1),
        Eigen::Vector3d(-1, 0, 0), Eigen::Vector3d( 0,-1, 0)
    };

    auto [cluster_assignments, final_seed_points, total_displacement, cluster_counts] =
        dynamic_seed_growth(eigen_normals, seed_points, params.kmeans_max_iterations, params.kmeans_tolerance);

    out.cluster_assignments = cluster_assignments;
    out.cluster_counts = cluster_counts;
    out.seed_displacements = total_displacement;
    out.quality = calculate_sphere_quality(total_displacement, cluster_counts,
                                           params.min_populated_seeds,
                                           params.min_seed_points,
                                           params.max_mean_displacement,
                                           params.max_std_displacement);
    out.accepted = out.quality.accepted;
    return true;
}

void saveFragmentSummaryCsv(const std::string& path, const std::vector<SSCFragmentResult>& results)
{
    std::ofstream out(path);
    if (!out.is_open()) return;
    out << "fragment_id,accepted,start_initial_frame,end_initial_frame,gps_t0,gps_t1,extent_m,populated_seeds,mean_displacement,std_displacement,num_points\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const size_t n = r.xyz ? r.xyz->size() : 0;
        out << i << "," << (r.accepted ? 1 : 0) << "," << r.start_initial_frame << "," << r.end_initial_frame << ","
            << r.gps_t0 << "," << r.gps_t1 << "," << r.extent_m << "," << r.quality.populated_seeds << ","
            << r.quality.mean_displacement << "," << r.quality.std_displacement << "," << n << "\n";
    }
}
