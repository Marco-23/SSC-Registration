/*
===============================================================================
SSC visualization helpers
-------------------------------------------------------------------------------
This file contains optional viewer utilities for inspecting point clouds,
normals, and fragment geometry during the SSC workflow.

Reference article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
The functions in this file are intended for visual debugging and interpretation,
for example:

- displaying a candidate fragment,
- visualizing estimated normals,
- inspecting centroid-based normal orientation,
- showing accepted or rejected fragment geometry.

Important note
--------------
Visualization is optional and depends on the local build configuration. These
functions are useful for debugging and understanding SSC behavior, but they are
not required for the core fragmentation logic.
===============================================================================
*/


#include "visualization_utils.h"

#include <array>
#include <iostream>
#include <pcl/io/ply_io.h>

#if defined(SSC_HAS_PCL_VIS) && SSC_HAS_PCL_VIS
  #include <pcl/visualization/pcl_visualizer.h>
#endif

void saveFragmentsPreviewPly(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& fragments, const std::string& path)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
    static const std::array<std::array<uint8_t,3>,12> palette{{
        {{230,57,70}}, {{29,185,84}}, {{66,135,245}}, {{255,196,0}},
        {{156,39,176}}, {{255,87,34}}, {{3,169,244}}, {{139,195,74}},
        {{233,30,99}}, {{0,188,212}}, {{121,85,72}}, {{158,158,158}}
    }};
    for (size_t i = 0; i < fragments.size(); ++i) {
        const auto& frag = fragments[i];
        if (!frag) continue;
        const auto& c = palette[i % palette.size()];
        for (const auto& p : frag->points) {
            pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z; q.r = c[0]; q.g = c[1]; q.b = c[2];
            out->push_back(q);
        }
    }
    out->width = static_cast<uint32_t>(out->size());
    out->height = 1;
    out->is_dense = false;
    if (pcl::io::savePLYFileBinary(path, *out) == 0) {
        std::cout << "Saved fragment preview PLY: " << path << "\n";
    }
}

void maybeShowFragmentsViewer(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& fragments, bool show)
{
#if defined(SSC_HAS_PCL_VIS) && SSC_HAS_PCL_VIS
    if (!show) return;
    static const std::array<std::array<double,3>,12> palette{{
        {{0.90,0.22,0.27}}, {{0.11,0.73,0.33}}, {{0.26,0.53,0.96}}, {{1.00,0.77,0.00}},
        {{0.61,0.15,0.69}}, {{1.00,0.34,0.13}}, {{0.01,0.66,0.96}}, {{0.55,0.76,0.29}},
        {{0.91,0.12,0.39}}, {{0.00,0.74,0.83}}, {{0.47,0.33,0.28}}, {{0.62,0.62,0.62}}
    }};
    auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("SSC fragments"));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);
    for (size_t i = 0; i < fragments.size(); ++i) {
        const auto& frag = fragments[i];
        if (!frag || frag->empty()) continue;
        const auto& c = palette[i % palette.size()];
        const std::string id = "frag_" + std::to_string(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col(
            frag, static_cast<int>(c[0] * 255.0), static_cast<int>(c[1] * 255.0), static_cast<int>(c[2] * 255.0));
        viewer->addPointCloud<pcl::PointXYZ>(frag, col, id);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, id);
    }
    while (!viewer->wasStopped()) viewer->spinOnce(100);
#else
    (void)fragments;
    if (show) {
        std::cerr << "Viewer requested, but PCL visualization is not available in this build.\n";
    }
#endif
}
