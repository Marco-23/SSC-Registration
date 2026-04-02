/*
===============================================================================
Semi-sphere Check (SSC) adaptive fragmentation demonstration
-------------------------------------------------------------------------------
Original implementation adapted from the repository/project workflow.

Related article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

What this program does
----------------------
This program performs the SSC-style adaptive fragmentation stage described in
the paper for MLS point clouds:

1. Load an MLS point cloud (.pcd or .ply).
2. Detect and read the GPS time field from the point cloud attributes.
3. Sort the points by GPS time.
4. Split the cloud into initial time-based frames.
5. Build candidate fragments from consecutive initial frames.
6. For each candidate fragment:
   - estimate normals on the whole current candidate,
   - orient normals consistently,
   - project normal directions to a semi-sphere,
   - cluster them using five canonical seed directions,
   - evaluate whether the fragment contains sufficiently distributed
     mutually orthogonal geometry.
7. If the candidate fragment is accepted, save it as an SSC fragment.
8. If the candidate fragment is rejected, append the next initial frame and
   test again.
9. Optionally force acceptance when the fragment extent exceeds the configured
   maximum extent threshold.

Why this is useful
------------------
Large MLS point clouds often suffer from trajectory drift over long distances.
Using one single rigid transformation for the full scan can therefore be
unreliable. SSC adaptively divides the MLS trajectory into fragments that are:

- small enough to reduce the effect of drift,
- but large enough to contain stable and diverse planar geometry for
  registration.

This makes the later registration stages more robust.


Command line usage
------------------
  ./ssc_full_pipeline <input_cloud(.pcd|.ply)> <output_directory> [options]

Example:
  ./ssc_full_pipeline source.ply out_dir --no-save-initial-frames --show

Typical optional arguments
--------------------------
  --split-seconds <value>
      Duration of the initial GPS-time split.

  --normal-k <value>
      Number of neighbors used for normal estimation.

  --min-fragment-extent <meters>
      Minimum candidate extent before it can be accepted by SSC.

  --max-extra-frames <value>
      Maximum number of additional initial frames that may be merged into the
      current candidate.

  --min-populated-seeds <value>
      Minimum number of sufficiently populated seed clusters required.

  --min-seed-points <value>
      Minimum number of normals assigned to a seed for it to count as populated.

  --max-mean-disp <value>
      Maximum allowed mean seed displacement for SSC acceptance.

  --max-std-disp <value>
      Maximum allowed standard deviation of seed displacement for SSC acceptance.

  --accept-if-extent-over <meters>
      Project-specific override: automatically accept a fragment once its
      extent exceeds this threshold.

  --show
      Request visualization if the build includes viewer support.
===============================================================================
*/


#include "file_io.h"
#include "ssc_pipeline.h"
#include "visualization_utils.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <pcl/io/pcd_io.h>

namespace fs = std::filesystem;

struct ProgramOptions {
    std::string input_path;
    std::string output_dir;
    SSCParams ssc;
};

static void printUsage(const char* exe)
{
    std::cout
        << "Usage:\n"
        << "  " << exe << " <input_cloud.(pcd|ply)> <output_dir> [options]\n\n"
        << "Options:\n"
        << "  --split-seconds <sec>       GPS split interval (default 10)\n"
        << "  --max-range <m>             Filter points with Range/range > value (default 20)\n"
        << "  --no-range-filter           Disable range filtering\n"
        << "  --normal-k <k>              K for normal estimation (default 300)\n"
        << "  --min-fragment-extent <m>   Minimum spatial extent before SSC check (default 10)\n"
        << "  --accept-if-extent-over <m> Auto-accept fragment above this extent (project rule)\n"
        << "  --max-extra-frames <n>      Maximum extra initial frames to append (default 10)\n"
        << "  --min-populated-seeds <n>   Minimum populated seeds (default 4)\n"
        << "  --min-seed-points <n>       Minimum points to count a seed as populated (default 1000)\n"
        << "  --max-mean-disp <v>         Mean seed displacement threshold (default 0.25)\n"
        << "  --max-std-disp <v>          Std seed displacement threshold (default 0.18)\n"
        << "  --show                      Show accepted fragments in a PCL viewer when available\n"
        << "  --no-save-initial-frames    Do not save GPS-split initial frames\n"
        << "  --no-preview-ply            Do not save the colored fragment preview PLY\n";
}

static bool parseArgs(int argc, char** argv, ProgramOptions& opt)
{
    if (argc < 3) return false;
    opt.input_path = argv[1];
    opt.output_dir = argv[2];

    for (int arg_index = 3; arg_index < argc; ++arg_index) {
        const std::string arg = argv[arg_index];
        auto needValue = [&](const char* name) -> const char* {
            if (arg_index + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + name);
            return argv[++arg_index];
        };
        if (arg == "--split-seconds") opt.ssc.split_seconds = std::stod(needValue("--split-seconds"));
        else if (arg == "--max-range") opt.ssc.max_range = std::stod(needValue("--max-range"));
        else if (arg == "--no-range-filter") opt.ssc.use_range_filter = false;
        else if (arg == "--normal-k") opt.ssc.normal_k = std::stoi(needValue("--normal-k"));
        else if (arg == "--min-fragment-extent") opt.ssc.min_fragment_extent_m = std::stod(needValue("--min-fragment-extent"));
        else if (arg == "--accept-if-extent-over") opt.ssc.accept_if_extent_over_m = std::stod(needValue("--accept-if-extent-over"));
        else if (arg == "--max-extra-frames") opt.ssc.max_extra_frames = std::stoi(needValue("--max-extra-frames"));
        else if (arg == "--min-populated-seeds") opt.ssc.min_populated_seeds = std::stoi(needValue("--min-populated-seeds"));
        else if (arg == "--min-seed-points") opt.ssc.min_seed_points = std::stoi(needValue("--min-seed-points"));
        else if (arg == "--max-mean-disp") opt.ssc.max_mean_displacement = std::stod(needValue("--max-mean-disp"));
        else if (arg == "--max-std-disp") opt.ssc.max_std_displacement = std::stod(needValue("--max-std-disp"));
        else if (arg == "--show") opt.ssc.show_viewer = true;
        else if (arg == "--no-save-initial-frames") opt.ssc.save_initial_frames = false;
        else if (arg == "--no-preview-ply") opt.ssc.save_preview_ply = false;
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    return true;
}

static pcl::PCLPointCloud2 reorderBlobByIndices(const pcl::PCLPointCloud2& src, const std::vector<int>& order)
{
    return sliceCloud2ByIndices(src, order);
}

static std::vector<int> sortIndicesByGpsAndOptionalRange(const LoadedCloud& loaded_cloud, const SSCParams& params)
{
    const size_t point_count = loaded_cloud.xyz ? loaded_cloud.xyz->size() : 0;
    std::vector<int> sorted_point_indices;
    sorted_point_indices.reserve(point_count);
    for (size_t point_index = 0; point_index < point_count; ++point_index) {
        if (params.use_range_filter && loaded_cloud.has_range && point_index < loaded_cloud.range.size() && loaded_cloud.range[point_index] > params.max_range) continue;
        sorted_point_indices.push_back(static_cast<int>(point_index));
    }
    std::stable_sort(sorted_point_indices.begin(), sorted_point_indices.end(), [&](int lhs, int rhs) {
        return loaded_cloud.gps_time[static_cast<size_t>(lhs)] < loaded_cloud.gps_time[static_cast<size_t>(rhs)];
    });
    return sorted_point_indices;
}

static bool sameFieldLayout(const std::vector<pcl::PCLPointField>& a, const std::vector<pcl::PCLPointField>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t field_index = 0; field_index < a.size(); ++field_index) {
        if (a[field_index].name != b[field_index].name) return false;
        if (a[field_index].offset != b[field_index].offset) return false;
        if (a[field_index].datatype != b[field_index].datatype) return false;
        if (a[field_index].count != b[field_index].count) return false;
    }
    return true;
}

static void appendFrameToBlob(pcl::PCLPointCloud2& candidate_blob, const pcl::PCLPointCloud2& next_frame_blob)
{
    if (candidate_blob.fields.empty()) {
        candidate_blob = next_frame_blob;
        return;
    }
    if (!sameFieldLayout(candidate_blob.fields, next_frame_blob.fields) || candidate_blob.point_step != next_frame_blob.point_step) {
        throw std::runtime_error("Cannot append initial frame with mismatched field layout.");
    }
    const size_t old_size = candidate_blob.data.size();
    candidate_blob.data.resize(old_size + next_frame_blob.data.size());
    std::copy(next_frame_blob.data.begin(), next_frame_blob.data.end(), candidate_blob.data.begin() + static_cast<std::ptrdiff_t>(old_size));
    candidate_blob.width += next_frame_blob.width;
    candidate_blob.height = 1;
    candidate_blob.row_step = candidate_blob.point_step * candidate_blob.width;
}

int main(int argc, char** argv)
{
    // High-level flow:
    // 1) load cloud
    // 2) detect GPS/range fields
    // 3) sort by GPS time
    // 4) build initial time frames
    // 5) adaptively merge frames and run SSC on each candidate fragment
    // 6) save accepted/rejected outputs and summary

    try {
        ProgramOptions options;
        if (!parseArgs(argc, argv, options)) { printUsage(argv[0]); return 1; }

        fs::create_directories(options.output_dir);
        fs::create_directories(fs::path(options.output_dir) / "initial_frames");
        fs::create_directories(fs::path(options.output_dir) / "ssc_fragments");
        fs::create_directories(fs::path(options.output_dir) / "rejected_fragments");

        LoadedCloud loaded_cloud;
        std::string error_message;
        if (!loadCloudGeneric(options.input_path, loaded_cloud, error_message)) throw std::runtime_error(error_message);

        std::cout << "Loaded input cloud\n"
                  << "  path: " << options.input_path << "\n"
                  << "  points: " << loaded_cloud.xyz->size() << "\n"
                  << "  has_gps_time: " << (loaded_cloud.has_gps_time ? "yes" : "no") << "\n"
                  << "  has_range: " << (loaded_cloud.has_range ? "yes" : "no") << "\n";

        if (!loaded_cloud.has_gps_time) throw std::runtime_error("Input cloud does not contain gps_time/Gps_Time. This standalone pipeline expects a point cloud with GPS time for initial splitting.");

        // Sort all retained points by GPS time first.
        // This creates the temporal order used by the initial frame split.
        const std::vector<int> sorted_point_indices = sortIndicesByGpsAndOptionalRange(loaded_cloud, options.ssc);
        if (sorted_point_indices.empty()) throw std::runtime_error("No points remain after filtering.");
        const pcl::PCLPointCloud2 time_sorted_blob = reorderBlobByIndices(*loaded_cloud.blob, sorted_point_indices);

        // Initial frames are simple GPS-time chunks.
        // SSC happens later on merged candidate fragments built from these units.
        std::vector<InitialFrame> initial_gps_frames;
        if (!buildInitialFramesFromGPS(time_sorted_blob, options.ssc.split_seconds, initial_gps_frames, error_message)) throw std::runtime_error(error_message);

        std::cout << "Initial GPS frames created: " << initial_gps_frames.size() << "\n";
        if (options.ssc.save_initial_frames) {
            for (const auto& gps_frame : initial_gps_frames) {
                const fs::path output_pcd_path = fs::path(options.output_dir) / "initial_frames" / (gps_frame.name + ".pcd");
                pcl::io::savePCDFile(output_pcd_path.string(), gps_frame.blob, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true);
            }
        }

        std::vector<SSCFragmentResult> fragment_results;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accepted_fragment_clouds;

        // Adaptive SSC loop:
        // start from one initial frame, then append more frames only when needed.
        size_t next_start_frame_index = 0;
        int output_fragment_id = 0;

        while (next_start_frame_index < initial_gps_frames.size()) {
            size_t current_end_frame_index = next_start_frame_index;
            int extra_frames_used = 0;

            auto candidate_fragment_xyz = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
            if (initial_gps_frames[next_start_frame_index].xyz) *candidate_fragment_xyz = *initial_gps_frames[next_start_frame_index].xyz;
            pcl::PCLPointCloud2 candidate_fragment_blob = initial_gps_frames[next_start_frame_index].blob;

            auto append_next_initial_frame = [&]() {
                ++current_end_frame_index;
                ++extra_frames_used;
                if (initial_gps_frames[current_end_frame_index].xyz) {
                    *candidate_fragment_xyz += *initial_gps_frames[current_end_frame_index].xyz;
                }
                appendFrameToBlob(candidate_fragment_blob, initial_gps_frames[current_end_frame_index].blob);
            };

            while (true) {
                // Each loop evaluates the WHOLE current candidate fragment.
                // If the candidate fails SSC, the next initial frame is appended,
                // and normals are recomputed for the entire merged fragment.
                const double candidate_extent_m = computeFragmentExtentMeters(candidate_fragment_xyz);
                const bool candidate_too_small_for_ssc = candidate_extent_m < options.ssc.min_fragment_extent_m;
                const bool can_still_grow = (current_end_frame_index + 1 < initial_gps_frames.size()) &&
                                            (extra_frames_used < options.ssc.max_extra_frames);

                if (candidate_too_small_for_ssc && can_still_grow) {
                    std::cout << "Candidate frames [" << next_start_frame_index << "," << current_end_frame_index
                              << "] extent=" << candidate_extent_m << " m is below the minimum SSC extent of "
                              << options.ssc.min_fragment_extent_m << " m. Appending the next initial frame.\n";
                    append_next_initial_frame();
                    continue;
                }

                SSCFragmentResult fragment_result;
                fragment_result.start_initial_frame = static_cast<int>(next_start_frame_index);
                fragment_result.end_initial_frame = static_cast<int>(current_end_frame_index);
                fragment_result.gps_t0 = initial_gps_frames[next_start_frame_index].t0;
                fragment_result.gps_t1 = initial_gps_frames[current_end_frame_index].t1;

                if (!runSSCOnFragment(candidate_fragment_xyz, options.ssc, fragment_result, error_message)) {
                    throw std::runtime_error(error_message);
                }

                if (!fragment_result.accepted && options.ssc.accept_if_extent_over_m > 0.0 && fragment_result.extent_m > options.ssc.accept_if_extent_over_m) {
                    fragment_result.accepted = true;
                }

                if (!fragment_result.accepted && can_still_grow) {
                    std::cout << "SSC rejected candidate frames [" << next_start_frame_index << "," << current_end_frame_index
                              << "]. Appending the next initial frame and repeating the SSC test.\n";
                    append_next_initial_frame();
                    continue;
                }

                // Finalize and save the fragment exactly as tested in this iteration.
                fragment_result.blob = candidate_fragment_blob;
                const fs::path output_directory = fs::path(options.output_dir) / (fragment_result.accepted ? "ssc_fragments" : "rejected_fragments");
                const fs::path output_pcd_path = output_directory / ("fragment_" + std::to_string(output_fragment_id) + ".pcd");
                pcl::io::savePCDFile(output_pcd_path.string(), fragment_result.blob, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true);

                std::cout << "Saved final fragment " << output_fragment_id
                          << " from initial frames [" << fragment_result.start_initial_frame << "," << fragment_result.end_initial_frame << "]"
                          << " | extent=" << fragment_result.extent_m << " m"
                          << " | accepted=" << (fragment_result.accepted ? "yes" : "no");
                if (options.ssc.accept_if_extent_over_m > 0.0 && fragment_result.extent_m > options.ssc.accept_if_extent_over_m) {
                    std::cout << " | acceptance_reason=max_extent_rule(" << options.ssc.accept_if_extent_over_m << " m)";
                } else {
                    std::cout << " | acceptance_reason=ssc_quality";
                }
                std::cout << " | output=" << output_pcd_path.string() << "\n";

                if (fragment_result.accepted) accepted_fragment_clouds.push_back(fragment_result.xyz);
                fragment_results.push_back(fragment_result);
                ++output_fragment_id;
                next_start_frame_index = current_end_frame_index + 1;
                break;
            }
        }

        saveFragmentSummaryCsv((fs::path(options.output_dir) / "ssc_summary.csv").string(), fragment_results);
        if (options.ssc.save_preview_ply && !accepted_fragment_clouds.empty()) {
            saveFragmentsPreviewPly(accepted_fragment_clouds, (fs::path(options.output_dir) / "ssc_fragments_preview.ply").string());
        }
        maybeShowFragmentsViewer(accepted_fragment_clouds, options.ssc.show_viewer);
        std::cout << "Pipeline finished. Accepted fragments: " << accepted_fragment_clouds.size()
                  << " / total final fragments: " << fragment_results.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
