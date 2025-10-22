#pragma once
#include <string>
#include <vector>
#include <cstddef>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>

#include "progress.hpp"  // TimingCollector

// Metadata for the GPS-time field inside a PCLPointCloud2 blob
struct GpsFieldInfo {
    int index       = -1;
    std::size_t offset = 0;
    int datatype    = -1;   // pcl::PCLPointField::datatype
    std::string name;
};

// Try to locate a GPS-time field in the cloud. We accept names that
// contain both "gps" and "time" (case-insensitive), allowing the
// CloudCompare "scalar_*" prefix, and types float32 / float64.
bool find_gps_time_field(const pcl::PCLPointCloud2& cloud, GpsFieldInfo& out);

// Bin the cloud’s points into contiguous frames of length `frameSec`,
// based on the detected GPS-time field. Returns counts per frame and
// the observed time range [tmin, tmax]. Points with NaN time are skipped.
bool fragment_frames_by_gps(const pcl::PCLPointCloud2& cloud,
                            const GpsFieldInfo& gps,
                            double frameSec,
                            std::vector<std::size_t>& counts,
                            double& tmin_out,
                            double& tmax_out);

// One-call convenience that (1) checks GPS time on the SOURCE,
// (2) fragments into frames, and (3) prints a concise summary.
// Timings are recorded into the given TimingCollector.
void run_fragmentation_step(const pcl::PCLPointCloud2& sourceClean,
                            double frameSec,
                            TimingCollector& tc);
