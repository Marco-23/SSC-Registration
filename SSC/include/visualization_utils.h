/*
===============================================================================
SSC visualization interface
-------------------------------------------------------------------------------
This header declares optional helper functions for visual inspection of the SSC
workflow.

Reference article:
  Ortiz Rincón, Yang, Holst (2026)
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
These declarations expose viewer-related utilities that can be used to inspect
candidate fragments, normals, and other intermediate SSC results.
===============================================================================
*/


#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

void saveFragmentsPreviewPly(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& fragments, const std::string& path);
void maybeShowFragmentsViewer(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& fragments, bool show);
