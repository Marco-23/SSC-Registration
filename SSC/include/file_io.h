/*
===============================================================================
SSC file I/O interface
-------------------------------------------------------------------------------
This header declares helper functions for reading and writing auxiliary files
used by the SSC implementation.

Reference article:
  Ortiz Rincón, Yang, Holst (2026)
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
These declarations support optional export and inspection of intermediate SSC
results such as normals, coordinates, and cluster labels.
===============================================================================
*/



#pragma once
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

struct LoadedCloud {
    pcl::PCLPointCloud2::Ptr blob;
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz;
    std::vector<double> gps_time;
    bool has_gps_time = false;
    std::vector<double> range;
    bool has_range = false;
};

bool loadCloudGeneric(const std::string& path, LoadedCloud& out, std::string& err);
int findFieldCI(const pcl::PCLPointCloud2& blob, const std::string& name);
bool readFieldAsDoubleVector(const pcl::PCLPointCloud2& blob, const std::string& name, std::vector<double>& out);
pcl::PCLPointCloud2 sliceCloud2ByIndices(const pcl::PCLPointCloud2& src, const std::vector<int>& indices);
void save_ascii_file(const std::vector<std::vector<float>>& coordinates,
                     const std::vector<std::vector<float>>& normals,
                     const std::vector<int>& cluster_labels,
                     const std::string& output_file_path);
