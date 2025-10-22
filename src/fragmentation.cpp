#include "fragmentation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>   // memcpy
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {
    std::string lower_copy(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return s;
    }

    bool is_gps_time_name(std::string n) {
        n = lower_copy(std::move(n));
        if (n.rfind("scalar_", 0) == 0) n = n.substr(7); // CloudCompare prefix
        return (n.find("gps") != std::string::npos) && (n.find("time") != std::string::npos);
    }

    const char* dtype_name(int dt) {
        using PF = pcl::PCLPointField;
        switch (dt) {
            case PF::INT8:    return "int8";
            case PF::UINT8:   return "uint8";
            case PF::INT16:   return "int16";
            case PF::UINT16:  return "uint16";
            case PF::INT32:   return "int32";
            case PF::UINT32:  return "uint32";
            case PF::FLOAT32: return "float32";
            case PF::FLOAT64: return "float64";
            default:          return "?";
        }
    }
}

bool find_gps_time_field(const pcl::PCLPointCloud2& cloud, GpsFieldInfo& out) {
    out = {};
    for (std::size_t i = 0; i < cloud.fields.size(); ++i) {
        const auto& f = cloud.fields[i];
        if (!is_gps_time_name(f.name)) continue;
        if (f.datatype == pcl::PCLPointField::FLOAT32 ||
            f.datatype == pcl::PCLPointField::FLOAT64) {
            out.index    = static_cast<int>(i);
            out.offset   = static_cast<std::size_t>(f.offset);
            out.datatype = f.datatype;
            out.name     = f.name;
            return true;
        }
    }
    return false;
}

bool fragment_frames_by_gps(const pcl::PCLPointCloud2& cloud,
                            const GpsFieldInfo& gps,
                            double frameSec,
                            std::vector<std::size_t>& counts,
                            double& tmin_out,
                            double& tmax_out)
{
    if (gps.index < 0 || frameSec <= 0.0) return false;

    const std::size_t H = (cloud.height == 0 ? 1 : static_cast<std::size_t>(cloud.height));
    const std::size_t N = static_cast<std::size_t>(cloud.width) * H;
    const std::size_t step = static_cast<std::size_t>(cloud.point_step);
    const auto* base = cloud.data.data();

    auto read_time = [&](const std::uint8_t* p)->double {
        if (gps.datatype == pcl::PCLPointField::FLOAT32) {
            float v;
            std::memcpy(&v, p + gps.offset, sizeof(float));
            return static_cast<double>(v);
        } else if (gps.datatype == pcl::PCLPointField::FLOAT64) {
            double v;
            std::memcpy(&v, p + gps.offset, sizeof(double));
            return v;
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    // Pass 1: min/max
    double tmin =  std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();

    const std::uint8_t* p = base;
    for (std::size_t i = 0; i < N; ++i, p += step) {
        const double t = read_time(p);
        if (t == t) { // not NaN
            if (t < tmin) tmin = t;
            if (t > tmax) tmax = t;
        }
    }
    if (!std::isfinite(tmin) || !std::isfinite(tmax) || tmax < tmin) return false;

    const double duration = tmax - tmin;
    std::size_t nFrames = (duration <= 0.0) ? 1 : static_cast<std::size_t>(std::ceil(duration / frameSec));
    if (nFrames == 0) nFrames = 1;
    counts.assign(nFrames, 0);

    // Pass 2: counts
    p = base;
    for (std::size_t i = 0; i < N; ++i, p += step) {
        const double t = read_time(p);
        if (t != t) continue;
        std::size_t idx = (t <= tmin) ? 0 : static_cast<std::size_t>((t - tmin) / frameSec);
        if (idx >= nFrames) idx = nFrames - 1;
        counts[idx]++;
    }

    tmin_out = tmin;
    tmax_out = tmax;
    return true;
}

void run_fragmentation_step(const pcl::PCLPointCloud2& sourceClean,
                            double frameSec,
                            TimingCollector& tc)
{
    // 1) Check GPS time
    GpsFieldInfo gps{};
    {
        auto t_chk = tc.scope("Check GPS time (source)");
        if (!find_gps_time_field(sourceClean, gps)) {
            std::cerr << "[ERROR] Source has no GPS time field; skipping fragmentation.\n";
            return;
        }
        std::cout << "[GPS] Source field detected: '" << gps.name
                  << "' (" << dtype_name(gps.datatype) << ")\n";
    }

    // 2) Fragment by GPS time and print summary
    {
        auto t_frag = tc.scope("Fragment source by 10 s frames");
        std::vector<std::size_t> counts;
        double tmin = 0.0, tmax = 0.0;

        if (!fragment_frames_by_gps(sourceClean, gps, frameSec, counts, tmin, tmax)) {
            std::cerr << "[ERROR] Fragmentation by GPS time failed.\n";
            return;
        }

        const double span = tmax - tmin;
        const std::size_t nF = counts.size();

        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(2);
        std::cout << "[Frames] GPS span: " << span << " s  |  frame size: "
                  << frameSec << " s  ->  " << nF << " frames\n";
        std::cout << std::setprecision(3)
                  << "         t_start=" << tmin << "   t_end=" << tmax << "\n";

        std::cout << "         counts per frame:";
        for (std::size_t i = 0; i < nF; ++i) {
            if (i % 10 == 0) std::cout << "\n           ";
            std::cout << "#" << i << "=" << counts[i] << " ";
        }
        std::cout << "\n";
        std::cout.unsetf(std::ios::fixed);
    }
}
