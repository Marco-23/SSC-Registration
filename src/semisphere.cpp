// semisphere.cpp — SSC with robust window pre-clip, robust PCA extent,
// per-frame normals (curvature-robust), axis-band aggregation across frames,
// world-frame robust BBOX acceptance, progress log, and accept-last-window.

#include "semisphere.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>

#include <pcl/conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

// --------------------- constants & small helpers ---------------------
template <typename T> static inline T clamp(T v, T lo, T hi) {
    return std::min(hi, std::max(lo, v));
}

// Robust trimming (percentiles)
static constexpr double CLIP_Q        = 0.005;  // 0.5% per-axis pre-clip for window/frame
static constexpr double CLIP_PAD_FRAC = 0.05;   // 5% padding beyond the clipped range
static constexpr double EXTENT_TRIM_Q = 0.02;   // 2%–98% span for PCA extent

// Only applies to legacy PCA-extent auto-accept path (kept for compatibility)
static constexpr bool LENGTH_ACCEPT_ONLY_AT_END = true;

static inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return char(std::tolower(c)); });
    return s;
}

static inline std::string norm_name(std::string s) {
    s = lower(s);
    s.erase(std::remove_if(s.begin(), s.end(), [](char c){ return c=='_'||c=='-'||c==' '; }), s.end());
    return s;
}

struct FieldInfo { int idx=-1; std::size_t offset=0; int datatype=0; std::string name; };

bool find_gps_field(const pcl::PCLPointCloud2& cloud, FieldInfo& out) {
    static const std::vector<std::string> candidates = {
        "gpstime","gpstimeoffset","gps_time","scalargpstime","scalar_gps_time","time","timestamp"
    };
    for (int i=0;i<static_cast<int>(cloud.fields.size());++i) {
        const auto& f = cloud.fields[i];
        std::string n = norm_name(f.name);
        for (auto& c : candidates) {
            if (n.find(c) != std::string::npos) {
                out.idx = i; out.offset = f.offset; out.datatype = f.datatype; out.name = f.name;
                return true;
            }
        }
    }
    return false;
}

bool read_time_at(const pcl::PCLPointCloud2& cloud, const FieldInfo& gps, std::size_t i, double& t) {
    const std::size_t npts = static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height ? cloud.height : 1);
    if (i>=npts) return false;
    const uint8_t* base = cloud.data.data() + i * cloud.point_step + gps.offset;
    switch (gps.datatype) {
        case pcl::PCLPointField::FLOAT32: { float v; std::memcpy(&v, base, sizeof(float)); t = static_cast<double>(v); return std::isfinite(t); }
        case pcl::PCLPointField::FLOAT64: { double v; std::memcpy(&v, base, sizeof(double)); t = v; return std::isfinite(t); }
        default: return false;
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr extract_xyz(const pcl::PCLPointCloud2& in) {
    // locate x,y,z
    const auto find_field = [&](const std::string& want)->FieldInfo{
        FieldInfo r;
        for (int i=0;i<static_cast<int>(in.fields.size());++i) {
            if (norm_name(in.fields[i].name) == want) {
                r.idx = i; r.offset = in.fields[i].offset; r.datatype = in.fields[i].datatype; r.name = in.fields[i].name;
                return r;
            }
        }
        return r;
    };
    FieldInfo fx = find_field("x");
    FieldInfo fy = find_field("y");
    FieldInfo fz = find_field("z");
    if (fx.idx<0||fy.idx<0||fz.idx<0 ||
        fx.datatype!=pcl::PCLPointField::FLOAT32||
        fy.datatype!=pcl::PCLPointField::FLOAT32||
        fz.datatype!=pcl::PCLPointField::FLOAT32) return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    const std::size_t npts = static_cast<std::size_t>(in.width) *
                             static_cast<std::size_t>(in.height ? in.height : 1);
    const std::size_t step = static_cast<std::size_t>(in.point_step);
    const std::size_t ox = static_cast<std::size_t>(fx.offset);
    const std::size_t oy = static_cast<std::size_t>(fy.offset);
    const std::size_t oz = static_cast<std::size_t>(fz.offset);

    const uint8_t* ptr = in.data.data();

    auto pc = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pc->resize(npts);
    for (std::size_t i=0;i<npts;++i) {
        float x,y,z;
        std::memcpy(&x, ptr + i*step + ox, sizeof(float));
        std::memcpy(&y, ptr + i*step + oy, sizeof(float));
        std::memcpy(&z, ptr + i*step + oz, sizeof(float));
        (*pc)[i].x = x; (*pc)[i].y = y; (*pc)[i].z = z;
    }
    return pc;
}

std::vector<int> subsample_indices(int N, int max_keep) {
    if (N <= max_keep) {
        std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0); return idx;
    }
    std::vector<int> all(N); std::iota(all.begin(), all.end(), 0);
    // deterministic shuffle (fixed seed)
    std::mt19937 rng(1234567);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(max_keep);
    std::sort(all.begin(), all.end());
    return all;
}

static Eigen::Matrix3d make_right_handed(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1) {
    Eigen::Vector3d x = v0.normalized();
    Eigen::Vector3d z = x.cross(v1).normalized();
    if (z.norm() == 0.0) {
        // pick arbitrary orthonormal axes
        z = (std::abs(x.x()) > 0.5 ? Eigen::Vector3d(0,1,0) : Eigen::Vector3d(1,0,0)).cross(x).normalized();
    }
    Eigen::Vector3d y = z.cross(x).normalized();
    Eigen::Matrix3d B;
    B.col(0)=x; B.col(1)=y; B.col(2)=z;
    return B;
}

struct AlignmentResult {
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned;
    Eigen::Matrix3d R;
    Eigen::Vector3d c;
    double extent_x = 0.0;
};

// ---------- robust percentile utility ----------
template <class It>
static inline double pctile_inplace(It begin, It end, double q) {
    const std::size_t n = static_cast<std::size_t>(end - begin);
    if (n == 0) return 0.0;
    const std::size_t idx = std::min<std::size_t>(n-1, static_cast<std::size_t>(std::floor(q * (n-1))));
    std::nth_element(begin, begin + idx, end);
    return static_cast<double>(*(begin + idx));
}



// Robust extents (2%–98%) in WORLD frame for a merged window (uniform subsample)
static std::tuple<double,double,double>
robust_world_extents(const pcl::PointCloud<pcl::PointXYZ>& all,
                     const std::vector<int>& idxs,
                     std::size_t max_sample = 200000)
{
    if (idxs.empty()) return {0.0,0.0,0.0};

    const std::size_t N = idxs.size();
    std::vector<float> xs, ys, zs;

    if (N <= max_sample) {
        xs.reserve(N); ys.reserve(N); zs.reserve(N);
        for (std::size_t k=0; k<N; ++k) {
            const auto& p = all[idxs[k]];
            xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z);
        }
    } else {
        // Uniform, deterministic subsample over [0, N)
        std::vector<int> pos = subsample_indices(static_cast<int>(N), static_cast<int>(max_sample));
        xs.reserve(pos.size()); ys.reserve(pos.size()); zs.reserve(pos.size());
        for (int k : pos) {
            const auto& p = all[idxs[static_cast<std::size_t>(k)]];
            xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z);
        }
    }

    auto X = xs, Y = ys, Z = zs;
    const double x_lo = pctile_inplace(X.begin(), X.end(), 0.02);
    const double x_hi = pctile_inplace(X.begin(), X.end(), 0.98);
    const double y_lo = pctile_inplace(Y.begin(), Y.end(), 0.02);
    const double y_hi = pctile_inplace(Y.begin(), Y.end(), 0.98);
    const double z_lo = pctile_inplace(Z.begin(), Z.end(), 0.02);
    const double z_hi = pctile_inplace(Z.begin(), Z.end(), 0.98);
    const double ex = std::max(0.0, x_hi - x_lo);
    const double ey = std::max(0.0, y_hi - y_lo);
    const double ez = std::max(0.0, z_hi - z_lo);
    return {ex, ey, ez};
}



// ---------- window/frame pre-clip (remove far spikes before PCA) ----------
static pcl::PointCloud<pcl::PointXYZ>::Ptr robust_clip_window_xyz(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
    if (!pc || pc->size() < 200) return pc;

    std::vector<float> xs, ys, zs;
    xs.reserve(pc->size()); ys.reserve(pc->size()); zs.reserve(pc->size());
    for (const auto& p : *pc) { xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z); }

    // compute per-axis clipped range with padding
    auto xs_copy = xs, ys_copy = ys, zs_copy = zs;

    const double x_lo = pctile_inplace(xs_copy.begin(), xs_copy.end(), CLIP_Q);
    const double x_hi = pctile_inplace(xs_copy.begin(), xs_copy.end(), 1.0 - CLIP_Q);
    const double y_lo = pctile_inplace(ys_copy.begin(), ys_copy.end(), CLIP_Q);
    const double y_hi = pctile_inplace(ys_copy.begin(), ys_copy.end(), 1.0 - CLIP_Q);
    const double z_lo = pctile_inplace(zs_copy.begin(), zs_copy.end(), CLIP_Q);
    const double z_hi = pctile_inplace(zs_copy.begin(), zs_copy.end(), 1.0 - CLIP_Q);

    const double pad_x = CLIP_PAD_FRAC * std::max(1e-6, x_hi - x_lo);
    const double pad_y = CLIP_PAD_FRAC * std::max(1e-6, y_hi - y_lo);
    const double pad_z = CLIP_PAD_FRAC * std::max(1e-6, z_hi - z_lo);

    const double X0 = x_lo - pad_x, X1 = x_hi + pad_x;
    const double Y0 = y_lo - pad_y, Y1 = y_hi + pad_y;
    const double Z0 = z_lo - pad_z, Z1 = z_hi + pad_z;

    auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    out->reserve(pc->size());
    std::size_t removed = 0;

    for (const auto& p : *pc) {
        if (p.x < X0 || p.x > X1 || p.y < Y0 || p.y > Y1 || p.z < Z0 || p.z > Z1) { ++removed; continue; }
        out->push_back(p);
    }

    if (removed > 0) {
        std::cerr << "[SSC] window pre-clip removed " << removed << " spike(s)\n";
    }
    return out->empty() ? pc : out; // never return empty
}

AlignmentResult pca_align_to_x(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_in, bool robust_time_flip,
                               const std::vector<double>* opt_times=nullptr,
                               const std::vector<int>* opt_indices=nullptr)
{
    if (!pc_in || pc_in->empty()) return {};

    // --- robust pre-clip (remove far spikes that don't represent the bulk of the set)
    auto pc = robust_clip_window_xyz(pc_in);

    Eigen::Vector4f c4; pcl::compute3DCentroid(*pc, c4);
    Eigen::Vector3d c(c4.x(), c4.y(), c4.z());
    Eigen::Matrix3d Cov = Eigen::Matrix3d::Zero();

    for (const auto& p : *pc) {
        Eigen::Vector3d v(p.x, p.y, p.z), d = v - c;
        Cov += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Cov);
    Eigen::Vector3d v0 = es.eigenvectors().col(2).normalized();
    Eigen::Vector3d v1 = es.eigenvectors().col(1).normalized();

    // ---- robust time-direction flip ----
    if (robust_time_flip && opt_times && opt_indices) {
        const std::size_t m = std::min<std::size_t>(opt_indices->size(), pc->size());
        std::vector<std::pair<double,double>> tproj;
        tproj.reserve(m);

        for (std::size_t k = 0; k < m; ++k) {
            const int j_global = (*opt_indices)[k];
            const auto& p = (*pc)[k];
            Eigen::Vector3d x(p.x,p.y,p.z);
            tproj.emplace_back((*opt_times)[j_global], (x - c).dot(v0));
        }

        std::sort(tproj.begin(), tproj.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });
        if (tproj.size() >= 10) {
            std::size_t q = std::max<std::size_t>(1u, tproj.size()/10);
            double s0 = 0, s1 = 0;
            for (std::size_t i=0;i<q;++i) s0 += tproj[i].second;
            for (std::size_t i=tproj.size()-q;i<tproj.size();++i) s1 += tproj[i].second;
            if (s1 < s0) v0 = -v0;
        }
    }

    Eigen::Matrix3d Bx = make_right_handed(v0, v1);
    Eigen::Matrix3d R  = Bx.transpose();

    auto aligned = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    aligned->reserve(pc->size());

    std::vector<double> xs; xs.reserve(pc->size());
    for (const auto& p : *pc) {
        Eigen::Vector3d x (p.x, p.y, p.z);
        Eigen::Vector3d xa = R * (x - c);
        aligned->emplace_back(static_cast<float>(xa.x()),
                              static_cast<float>(xa.y()),
                              static_cast<float>(xa.z()));
        xs.push_back(xa.x());
    }

    // robust (trimmed) extent: [2%, 98%] percentile span; fallback on tiny sets
    double extent_x = 0.0;
    if (xs.size() >= 50) {
        auto cp = xs;
        const double lo = pctile_inplace(cp.begin(), cp.end(), EXTENT_TRIM_Q);
        const double hi = pctile_inplace(cp.begin(), cp.end(), 1.0 - EXTENT_TRIM_Q);
        extent_x = std::max(0.0, hi - lo);
    } else {
        auto [min_it, max_it] = std::minmax_element(xs.begin(), xs.end());
        extent_x = (xs.empty() ? 0.0 : std::max(0.0, *max_it - *min_it));
    }

    return { aligned, R, c, extent_x };
}

void orient_normals(std::vector<Eigen::Vector3d>& N, const pcl::PointCloud<pcl::PointXYZ>& aligned_pc) {
    Eigen::Vector4f c4; pcl::compute3DCentroid(aligned_pc, c4);
    Eigen::Vector3d c(c4.x(), c4.y(), c4.z());
    for (std::size_t i=0;i<N.size();++i) {
        const auto& p = aligned_pc.points[i];
        Eigen::Vector3d v = Eigen::Vector3d(p.x, p.y, p.z) - c;
    if (N[i].dot(v) < 0.0) N[i] = -N[i];
    }
}

// ---------------- normals on aligned fragment ----------------
static constexpr std::size_t MAX_NORMALS = 120000; // cap normals for speed

std::vector<Eigen::Vector3d>
compute_normals_on_aligned(const pcl::PointCloud<pcl::PointXYZ>::Ptr& aligned_pc,
                           int knn)
{
    std::vector<Eigen::Vector3d> Ns;
    if (!aligned_pc || aligned_pc->empty()) return Ns;

    // Uniform subsample to ≤ MAX_NORMALS
    std::vector<int> keep = subsample_indices(static_cast<int>(aligned_pc->size()),
                                              static_cast<int>(MAX_NORMALS));
    auto pc_use = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pc_use->reserve(keep.size());
    for (int idx : keep) pc_use->push_back((*aligned_pc)[idx]);

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(pc_use);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);
    ne.setKSearch(std::max(3, knn));
    pcl::PointCloud<pcl::Normal> cloud_normals;
    ne.compute(cloud_normals);

    Ns.resize(cloud_normals.size());
    for (std::size_t i=0;i<cloud_normals.size();++i) {
        const auto& q = cloud_normals[i];
        Eigen::Vector3d n(q.normal_x, q.normal_y, q.normal_z);
        double L = n.norm();
        if (L>0) n /= L;
        Ns[i] = n;
    }

    // Orient outwards (relative to fragment centroid)
    orient_normals(Ns, *pc_use);

    // Fold to upper hemisphere (treat up/down same)
    for (auto& n : Ns) if (n.z() < 0.0) n.z() = -n.z();

    return Ns;
}

// ---------------- Axis-band counting (no k-means) ----------------
struct AxisBandCounts {
    std::uint64_t x_pos=0, x_neg=0, y_pos=0, y_neg=0, z_cap=0;
    std::uint64_t sum_x() const { return x_pos + x_neg; }
    std::uint64_t sum_y() const { return y_pos + y_neg; }
};
// static constexpr float AXIS_EQ_ABS_NZ_MAX = 0.342f;     // |nz| <= sin(20°)
// static constexpr float AXIS_DIR_COS       = 0.8660254f; // |nx| or |ny| >= cos(30°)
// static constexpr float AXIS_Z_COS         = 0.9063078f; // nz >= cos(25°)
// static constexpr std::uint64_t AXIS_MIN_SUM = 200000;   // require ≥200k in X sum, Y sum, Z cap
// ---------------- Axis-band config (angles + thresholds) ----------------

// Helper without relying on M_PI
static inline float deg2rad(float d) { return d * 3.14159265358979323846f / 180.0f; }

// Configure half-angles (degrees). Smaller = tighter cones/bands.
static const float AXIS_EQ_HALF_DEG     = 15.0f; // equator band (± around horizon)
static const float AXIS_DIR_HALF_DEG_X  = 25.0f; // cone half-angle around ±X
static const float AXIS_DIR_HALF_DEG_Y  = 15.0f; // cone half-angle around ±Y (tighter than X)
static const float AXIS_Z_HALF_DEG      = 20.0f; // cone half-angle around +Z

// Derived numeric thresholds
static const float AXIS_EQ_ABS_NZ_MAX = std::sin(deg2rad(AXIS_EQ_HALF_DEG));     // |nz| <= sin(phi)
static const float AXIS_DIR_COS_X     = std::cos(deg2rad(AXIS_DIR_HALF_DEG_X));  // |nx| >= cos(theta_x)
static const float AXIS_DIR_COS_Y     = std::cos(deg2rad(AXIS_DIR_HALF_DEG_Y));  // |ny| >= cos(theta_y)
static const float AXIS_Z_COS         = std::cos(deg2rad(AXIS_Z_HALF_DEG));      //  nz  >= cos(theta_z)

// Axis-band vote requirement (scaled counts). Keep as-is unless you want earlier acceptance.
static constexpr std::uint64_t AXIS_MIN_SUM = 100000;   // need ≥200k in X sum, Y sum, and Z cap



static inline float fAbs_(float v){ return v<0?-v:v; }

static AxisBandCounts count_axis_bands(const std::vector<Eigen::Vector3d>& Ns) {
    AxisBandCounts C;
    for (const auto& n0 : Ns) {
        Eigen::Vector3d n = n0;
        if (n.z() < 0.0) n.z() = -n.z();
        const float nx = static_cast<float>(n.x());
        const float ny = static_cast<float>(n.y());
        const float nz = static_cast<float>(n.z());
        const float ax = fAbs_(nx), ay = fAbs_(ny), az = fAbs_(nz);

        // Z cap
        if (nz >= AXIS_Z_COS) ++C.z_cap;


        // Equator band & exclusive lateral axis
        if (az <= AXIS_EQ_ABS_NZ_MAX) {
            if (ax >= ay && ax >= AXIS_DIR_COS_X) {
                (nx >= 0.f) ? ++C.x_pos : ++C.x_neg;
            } else if (ay > ax && ay >= AXIS_DIR_COS_Y) {
                (ny >= 0.f) ? ++C.y_pos : ++C.y_neg;
            }
        }



    }
    return C;
}

// ---------------- SSC main ----------------
bool run_semi_sphere_check(const pcl::PCLPointCloud2& source_with_gps,
                           const SSCOptions& opt,
                           std::vector<SSCFragment>& out)
{
    out.clear();

    // 1) Extract xyz and GPS time
    FieldInfo gps;
    if (!find_gps_field(source_with_gps, gps)) {
        std::cerr << "[SSC] No GPS/time-like field found; skipping SSC.\n";
        return false;
    }

    auto all_xyz = extract_xyz(source_with_gps);
    const std::size_t N = all_xyz ? all_xyz->size() : 0;
    if (N == 0) {
        std::cerr << "[SSC] Source cloud empty.\n";
        return false;
    }

    // Gather all times
    std::vector<double> times(N, std::numeric_limits<double>::quiet_NaN());
    double tmin =  std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();
    for (std::size_t i=0;i<N;++i) {
        double t;
        if (read_time_at(source_with_gps, gps, i, t)) {
            times[i] = t;
            tmin = std::min(tmin, t); tmax = std::max(tmax, t);
        }
    }
    if (!(tmax > tmin)) { std::cerr << "[SSC] Invalid GPS time span.\n"; return false; }

    const int num_frames = std::max(1, static_cast<int>(std::ceil((tmax - tmin) / opt.frame_sec)));

    std::vector<std::vector<int>> frame_indices(num_frames);
    for (std::size_t i=0;i<N;++i) {
        double t = times[i]; if (!std::isfinite(t)) continue;
        int f = static_cast<int>(std::floor((t - tmin) / opt.frame_sec));
        f = clamp(f, 0, num_frames-1);
        frame_indices[f].push_back(static_cast<int>(i));
    }

    auto make_original_cloud = [&](const std::vector<int>& idxs)->pcl::PointCloud<pcl::PointXYZ>::Ptr {
        auto pc = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pc->reserve(idxs.size());
        for (int j : idxs) pc->push_back((*all_xyz)[j]);
        return pc;
    };

    // 2) Precompute per-frame axis-band tallies in each frame's local PCA frame
    struct FrameStat {
        std::size_t   npts  = 0;
        std::uint64_t xsum  = 0, ysum = 0, zcap = 0; // scaled to full frame
        std::size_t   nnorms= 0;                     // for reporting
    };
    std::vector<FrameStat> stats(num_frames);

    for (int f = 0; f < num_frames; ++f) {
        const auto& idxs = frame_indices[f];
        FrameStat fs; fs.npts = idxs.size();
        if (idxs.size() >= static_cast<std::size_t>(opt.min_points_for_eval)) {
            auto pc_raw = make_original_cloud(idxs);
            // local alignment (robust pre-clip inside)
            auto A      = pca_align_to_x(pc_raw, /*robust_time_flip*/false);
            auto Ns     = compute_normals_on_aligned(A.aligned, opt.normal_knn);
            AxisBandCounts ab = count_axis_bands(Ns);
            const double s = Ns.empty() ? 0.0 : static_cast<double>(idxs.size())/static_cast<double>(Ns.size());
            fs.xsum   = static_cast<std::uint64_t>(std::llround(s * (ab.x_pos + ab.x_neg)));
            fs.ysum   = static_cast<std::uint64_t>(std::llround(s * (ab.y_pos + ab.y_neg)));
            fs.zcap   = static_cast<std::uint64_t>(std::llround(s *  ab.z_cap));
            fs.nnorms = Ns.size();
        }
        stats[f] = fs;
    }

    bool any_ok = false;

    // 3) Sliding/growing windows (sum per-frame tallies; bbox in WORLD frame)
    for (int f0 = 0; f0 < num_frames; ) {
        int f1 = f0; // inclusive
        std::vector<int> merged = frame_indices[f0];
        double win_tmin = tmin + f0 * opt.frame_sec;
        double win_tmax = tmin + (f0 + 1) * opt.frame_sec;

        std::uint64_t Xsum=0, Ysum=0, Zcap=0;
        std::size_t   Npts = merged.size();
        std::size_t   Nnorms_sum = 0; // for reporting

        SSCFragment frag;
        frag.start_frame = f0; frag.end_frame = f1;
        frag.tmin = win_tmin;  frag.tmax  = win_tmax;

        bool decided = false;
        while (!decided) {
            // accumulate current frame stats
            const auto& fs = stats[f1];
            Xsum += fs.xsum; Ysum += fs.ysum; Zcap += fs.zcap;
            Nnorms_sum += fs.nnorms;

            // Evaluate acceptance
            const bool pass_axis = (Xsum >= AXIS_MIN_SUM &&
                                    Ysum >= AXIS_MIN_SUM &&
                                    Zcap >= AXIS_MIN_SUM);

            bool pass_bbox = false;
            double ex=0,ey=0,ez=0, emax=0;
            if (Npts >= static_cast<std::size_t>(opt.min_points_for_eval)) {
                std::tie(ex,ey,ez) = robust_world_extents(*all_xyz, merged);
                emax = std::max({ex,ey,ez});
                pass_bbox = (emax >= opt.accept_if_extent_over_m);
            }

            // Progress log
            std::cerr << "[SSC bins] frames[" << f0 << "," << f1 << "] "
                      << "Npts=" << Npts
                      << "  X~" << Xsum << " Y~" << Ysum << " Z~" << Zcap
                      << "  BBOX(" << std::setprecision(3) << ex << "," << ey << "," << ez << ")"
                      << " -> " << (pass_axis || pass_bbox ? "PASS" : "keep growing") << "\n";

            if (pass_axis || pass_bbox) {
                // build cloud & (optionally) bake window PCA for saving
                auto pc_raw = make_original_cloud(merged);
                if (opt.bake_alignment_into_xyz) {
                    auto Awin = pca_align_to_x(pc_raw, /*robust_time_flip*/opt.enforce_forward_time_increasing_x,
                                               &times, &merged);
                    frag.xyz = Awin.aligned;
                    frag.principal_extent_m = (pass_bbox ? emax : Awin.extent_x); // prefer world extent if used
                } else {
                    frag.xyz = pc_raw;
                    frag.principal_extent_m = (pass_bbox ? emax
                                                         : std::get<0>(robust_world_extents(*all_xyz, merged)));
                    auto [wx,wy,wz] = robust_world_extents(*all_xyz, merged);
                    frag.principal_extent_m = std::max({wx,wy,wz});
                }
                frag.start_frame = f0; frag.end_frame = f1;
                frag.tmin = win_tmin;  frag.tmax = win_tmax;
                // seed counts (split X,Y halves for reporting parity)
                frag.seed_counts = {
                    static_cast<int>(Xsum/2), static_cast<int>(Xsum - Xsum/2),
                    static_cast<int>(Ysum/2), static_cast<int>(Ysum - Ysum/2),
                    static_cast<int>(Zcap)
                };
                frag.normals_used = static_cast<int>(Nnorms_sum);
                frag.final_seeds.clear();
                frag.seed_disp_mean = 0.0;
                frag.seed_disp_std  = 0.0;
                // populated bins (rough reporting)
                const std::uint64_t bin_min = AXIS_MIN_SUM / 3;
                int valid = 0; for (int c : frag.seed_counts) if (static_cast<std::uint64_t>(c) >= bin_min) ++valid;
                frag.populated_seeds = valid;

                frag.accepted = true;
                any_ok = true;
                decided = true;
            } else {
                // grow if allowed
                
                if ((f1 + 1) < num_frames && (f1 - f0) < opt.max_extra_frames) {
                    ++f1;
                    merged.insert(merged.end(), frame_indices[f1].begin(), frame_indices[f1].end());
                    Npts += frame_indices[f1].size();
                    win_tmax = tmin + (f1 + 1) * opt.frame_sec;
                    continue;
                }
                // Can't grow: force-accept last window at end of sequence
                // Can't grow: force-accept last window at end of sequence
                // Can't grow: force-accept last window at end of sequence
                if ((f1 + 1) >= num_frames) {
                    auto pc_raw = make_original_cloud(merged);
                    if (opt.bake_alignment_into_xyz) {
                        auto Awin = pca_align_to_x(pc_raw, /*robust*/opt.enforce_forward_time_increasing_x,
                                                   &times, &merged);
                        frag.xyz = Awin.aligned;
                    } else {
                        frag.xyz = pc_raw;
                    }
                    auto [wx,wy,wz] = robust_world_extents(*all_xyz, merged);
                    frag.principal_extent_m = std::max({wx,wy,wz});
                
                    frag.start_frame = f0; frag.end_frame = f1;
                    frag.tmin = win_tmin;  frag.tmax  = win_tmax;
                
                    // ---- Fill stats so summary printing is safe ----
                    frag.seed_counts = {
                        static_cast<int>(Xsum/2), static_cast<int>(Xsum - Xsum/2),
                        static_cast<int>(Ysum/2), static_cast<int>(Ysum - Ysum/2),
                        static_cast<int>(Zcap)
                    };
                    frag.normals_used   = static_cast<int>(Nnorms_sum);
                    frag.final_seeds.clear();
                    frag.seed_disp_mean = 0.0;
                    frag.seed_disp_std  = 0.0;
                    const std::uint64_t bin_min = AXIS_MIN_SUM / 3;
                    int valid = 0; for (int c : frag.seed_counts) if (static_cast<std::uint64_t>(c) >= bin_min) ++valid;
                    frag.populated_seeds = valid;
                    // -----------------------------------------------
                
                    frag.accepted = true;
                    any_ok = true;
                    std::cerr << "[SSC] FORCE-ACCEPT (last window) frames[" << f0 << "," << f1
                              << "] Npts=" << Npts << " emax=" << frag.principal_extent_m << "\n";
                } else {
                    frag.accepted = false;
                }

                decided = true;
            }
        } // while

        if (!frag.xyz) frag.xyz.reset(new pcl::PointCloud<pcl::PointXYZ>);
        out.push_back(std::move(frag));

        // Advance start frame
        f0 = f1 + 1;
    }

    return any_ok;
}

void print_ssc_summary(const std::vector<SSCFragment>& frags, bool onlyAccepted)
{
    using std::cout;
    cout << "[SSC] Semi-sphere check summary (frames merged / acceptance)\n";
    cout << " idx | frames       | accepted | seeds | seed-std  | extent(m)\n";
    cout << "-----+--------------+----------+-------+-----------+----------\n";
    for (std::size_t i=0;i<frags.size();++i) {
        const auto& f = frags[i];
        if (onlyAccepted && !f.accepted) continue;
        cout << " " << std::setw(3) << i
             << " | [" << std::setw(2) << f.start_frame << "," << std::setw(2) << f.end_frame << "]"
             << " | " << (f.accepted ? "YES" : " no")
             << "     |   " << f.populated_seeds
             << "   |  " << std::fixed << std::setprecision(3) << f.seed_disp_std
             << "   | " << std::defaultfloat << f.principal_extent_m << "\n";

        if (f.accepted) {
            cout << "       seed-counts (est.): "
                 << "[+X:" << f.seed_counts[0]
                 << " -X:" << f.seed_counts[1]
                 << " +Y:" << f.seed_counts[2]
                 << " -Y:" << f.seed_counts[3]
                 << " +Z:" << f.seed_counts[4]
                 << "]  (Nnormals=" << f.normals_used << ")\n";
        }
        cout << std::defaultfloat;
    }
}

void print_ssc_summary(const std::vector<SSCFragment>& frags) {
    print_ssc_summary(frags, /*onlyAccepted=*/false);
}
