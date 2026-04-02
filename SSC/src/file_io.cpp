/*
===============================================================================
SSC file I/O helpers
-------------------------------------------------------------------------------
This file contains helper routines for reading and writing auxiliary data used
during the SSC workflow.

Reference article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

Purpose
-------
Typical responsibilities of this file include:

- reading exported coordinate/normal data,
- writing debugging or inspection files,
- saving cluster labels,
- exporting intermediate SSC results for external inspection.

===============================================================================
*/


#include "file_io.h"
#include "utilities.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <pcl/PCLPointField.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <sstream>

static std::string normalizeFieldName(std::string s)
{
    std::string out;
    out.reserve(s.size());
    for (unsigned char ch : s) {
        if (std::isalnum(ch)) out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

static bool fieldLooksLike(const std::string& field_name, const std::string& wanted)
{
    const std::string f = normalizeFieldName(field_name);
    const std::string w = normalizeFieldName(wanted);
    if (f == w) return true;

    // CloudCompare/PLY scalar fields often arrive with prefixes like scalar_, scalar-, scalar<space>
    // or with spaces/underscores removed by the reader. Accept robust substring matches.
    if (w == "gpstime") {
        return f.find("gpstime") != std::string::npos ||
               (f.find("gps") != std::string::npos && f.find("time") != std::string::npos);
    }
    if (w == "range") {
        return f.find("range") != std::string::npos;
    }
    return f.find(w) != std::string::npos;
}

int findFieldCI(const pcl::PCLPointCloud2& blob, const std::string& name)
{
    const std::string want_lower = toLowerCopy(name);
    for (int i = 0; i < static_cast<int>(blob.fields.size()); ++i) {
        if (toLowerCopy(blob.fields[static_cast<size_t>(i)].name) == want_lower) return i;
    }
    for (int i = 0; i < static_cast<int>(blob.fields.size()); ++i) {
        if (fieldLooksLike(blob.fields[static_cast<size_t>(i)].name, name)) return i;
    }
    return -1;
}

static size_t bytesPerType(uint8_t dt)
{
    using PF = pcl::PCLPointField;
    switch (dt) {
        case PF::INT8:
        case PF::UINT8:   return 1;
        case PF::INT16:
        case PF::UINT16:  return 2;
        case PF::INT32:
        case PF::UINT32:
        case PF::FLOAT32: return 4;
        case PF::FLOAT64:
        case PF::INT64:
        case PF::UINT64:  return 8;
        default: return 0;
    }
}

static bool readScalarAt(const uint8_t* p, uint8_t datatype, double& out)
{
    using PF = pcl::PCLPointField;
    switch (datatype) {
        case PF::INT8:    out = *reinterpret_cast<const int8_t*>(p); return true;
        case PF::UINT8:   out = *reinterpret_cast<const uint8_t*>(p); return true;
        case PF::INT16:   { int16_t v;  std::memcpy(&v, p, 2); out = v; return true; }
        case PF::UINT16:  { uint16_t v; std::memcpy(&v, p, 2); out = v; return true; }
        case PF::INT32:   { int32_t v;  std::memcpy(&v, p, 4); out = v; return true; }
        case PF::UINT32:  { uint32_t v; std::memcpy(&v, p, 4); out = v; return true; }
        case PF::FLOAT32: { float v;    std::memcpy(&v, p, 4); out = v; return true; }
        case PF::FLOAT64: { double v;   std::memcpy(&v, p, 8); out = v; return true; }
        case PF::INT64:   { int64_t v;  std::memcpy(&v, p, 8); out = static_cast<double>(v); return true; }
        case PF::UINT64:  { uint64_t v; std::memcpy(&v, p, 8); out = static_cast<double>(v); return true; }
        default: return false;
    }
}

bool readFieldAsDoubleVector(const pcl::PCLPointCloud2& blob, const std::string& name, std::vector<double>& out)
{
    const int idx = findFieldCI(blob, name);
    if (idx < 0) return false;
    const auto& f = blob.fields[static_cast<size_t>(idx)];
    const size_t b = bytesPerType(f.datatype);
    if (b == 0 || static_cast<size_t>(f.offset) + b > blob.point_step) return false;

    const size_t n = static_cast<size_t>(blob.width) * static_cast<size_t>(blob.height);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* row = blob.data.data() + i * blob.point_step + f.offset;
        if (!readScalarAt(row, f.datatype, out[i])) return false;
    }
    return true;
}

static std::string lowerExt(const std::string& path)
{
    const auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return {};
    return toLowerCopy(path.substr(pos));
}

static std::string buildFieldList(const pcl::PCLPointCloud2& blob)
{
    std::ostringstream oss;
    for (size_t i = 0; i < blob.fields.size(); ++i) {
        if (i) oss << ", ";
        oss << blob.fields[i].name;
    }
    return oss.str();
}

bool loadCloudGeneric(const std::string& path, LoadedCloud& out, std::string& err)
{
    out.blob.reset(new pcl::PCLPointCloud2);
    int rc = -1;
    const std::string ext = lowerExt(path);
    if (ext == ".pcd") rc = pcl::io::loadPCDFile(path, *out.blob);
    else if (ext == ".ply") rc = pcl::io::loadPLYFile(path, *out.blob);
    else { err = "Unsupported file extension. Supported: .pcd, .ply"; return false; }
    if (rc < 0) { err = "Failed to load point cloud: " + path; return false; }

    out.xyz.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromPCLPointCloud2(*out.blob, *out.xyz);
    if (!out.xyz || out.xyz->empty()) { err = "Loaded point cloud is empty."; return false; }

    out.has_gps_time = readFieldAsDoubleVector(*out.blob, "gps_time", out.gps_time) ||
                       readFieldAsDoubleVector(*out.blob, "Gps_Time", out.gps_time) ||
                       readFieldAsDoubleVector(*out.blob, "Gps Time", out.gps_time) ||
                       readFieldAsDoubleVector(*out.blob, "scalar Gps Time", out.gps_time) ||
                       readFieldAsDoubleVector(*out.blob, "time", out.gps_time);
    out.has_range = readFieldAsDoubleVector(*out.blob, "range", out.range) ||
                    readFieldAsDoubleVector(*out.blob, "Range", out.range) ||
                    readFieldAsDoubleVector(*out.blob, "scalar Range", out.range);

    std::cout << "Detected fields: " << buildFieldList(*out.blob) << "\n";
    if (out.has_gps_time) {
        const int gps_idx = findFieldCI(*out.blob, "gps_time");
        if (gps_idx >= 0) std::cout << "Using GPS time field: " << out.blob->fields[static_cast<size_t>(gps_idx)].name << "\n";
    }
    if (out.has_range) {
        const int range_idx = findFieldCI(*out.blob, "range");
        if (range_idx >= 0) std::cout << "Using range field: " << out.blob->fields[static_cast<size_t>(range_idx)].name << "\n";
    }

    return true;
}

pcl::PCLPointCloud2 sliceCloud2ByIndices(const pcl::PCLPointCloud2& src, const std::vector<int>& indices)
{
    pcl::PCLPointCloud2 out;
    out.header       = src.header;
    out.fields       = src.fields;
    out.is_bigendian = src.is_bigendian;
    out.point_step   = src.point_step;
    out.is_dense     = src.is_dense;

    const size_t N = static_cast<size_t>(src.width) * static_cast<size_t>(src.height);
    std::vector<int> valid;
    valid.reserve(indices.size());
    for (int i : indices) if (i >= 0 && static_cast<size_t>(i) < N) valid.push_back(i);

    out.height   = 1;
    out.width    = static_cast<uint32_t>(valid.size());
    out.row_step = out.point_step * out.width;
    out.data.resize(static_cast<size_t>(out.row_step));

    for (size_t j = 0; j < valid.size(); ++j) {
        const size_t si = static_cast<size_t>(valid[j]);
        std::memcpy(&out.data[j * out.point_step], &src.data[si * src.point_step], src.point_step);
    }
    return out;
}

void save_ascii_file(const std::vector<std::vector<float>>& coordinates,
                     const std::vector<std::vector<float>>& normals,
                     const std::vector<int>& cluster_labels,
                     const std::string& output_file_path)
{
    if (coordinates.size() != normals.size() || coordinates.size() != cluster_labels.size()) {
        std::cerr << "Error: Size mismatch while saving " << output_file_path << "\n";
        return;
    }
    std::ofstream file(output_file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << output_file_path << "\n";
        return;
    }
    file << "x,y,z,xn,yn,zn,cluster\n";
    for (size_t i = 0; i < coordinates.size(); ++i) {
        const auto& coord = coordinates[i];
        const auto& normal = normals[i];
        file << coord[0] << "," << coord[1] << "," << coord[2] << ","
             << normal[0] << "," << normal[1] << "," << normal[2] << ","
             << cluster_labels[i] << "\n";
    }
}
