/*
===============================================================================
PlaneBasedGICP / PV-GICP demonstration
-------------------------------------------------------------------------------
Original implementation written by the repository author.

Related article:
  "Quality-controlled registration of urban MLS point clouds reducing drift
  effects by adaptive fragmentation"
  Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
  International Journal of Applied Earth Observation and Geoinformation
  149 (2026) 105272
  DOI: 10.1016/j.jag.2026.105272

What this program does
----------------------
This program performs the planar-voxel selection and fine registration stage
used in a PV-GICP-style workflow:

1. Load a source cloud and a target cloud (.pcd or .ply).
2. Compute surface normals for both clouds.
3. Split the shared 3D space into fixed-size boxes (voxels).
4. Keep only the boxes that look planar in both clouds.
5. Aggregate those planar points into two reduced point clouds.
6. Run Generalized ICP (GICP) on the reduced planar subsets.

Why this is useful
------------------
Urban scenes usually contain many stable planar structures such as roads,
facades, and walls. By keeping only planar regions, the registration step can
be more robust and faster than using the full point cloud.

Command line usage
------------------
  ./gicp <source_cloud(.pcd|.ply)> <target_cloud(.pcd|.ply)> <box_size_in_meters>

Example:
  ./gicp source.ply target.pcd 2.0
===============================================================================
*/

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/search/kdtree.h>

namespace
{
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using PointCloudPtr = PointCloudT::Ptr;

using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using NormalCloudPtr = NormalCloudT::Ptr;

// Stores the main user-adjustable parameters of the algorithm.
struct Parameters
{
    std::size_t min_points_per_box = 50;
    float angle_threshold_degrees = 20.0f;
    float consistency_threshold = 0.70f;
    int k_search = 50;

    int gicp_max_iterations = 500;
    double gicp_max_correspondence_distance = 0.02;
    double gicp_transformation_epsilon = 1e-5;
    double gicp_euclidean_fitness_epsilon = 1e-5;
    bool gicp_use_reciprocal_correspondences = true;
};

// Simple axis-aligned bounding box.
struct BoundingBox
{
    PointT min_pt;
    PointT max_pt;
};

// Describes the voxel grid used for planarity selection.
struct GridDefinition
{
    float size_x = 1.0f;
    float size_y = 1.0f;
    float size_z = 1.0f;
    int divisions_x = 1;
    int divisions_y = 1;
    int divisions_z = 1;
    int total_boxes = 1;
};

// Collects the per-voxel points and normals for both source and target clouds.
struct BoxCollections
{
    std::vector<PointCloudPtr> source_boxes;
    std::vector<PointCloudPtr> target_boxes;
    std::vector<NormalCloudPtr> source_normal_boxes;
    std::vector<NormalCloudPtr> target_normal_boxes;
};

// Output of the planarity-selection stage.
struct PlanarSelectionResult
{
    PointCloudPtr planar_source{new PointCloudT};
    PointCloudPtr planar_target{new PointCloudT};
    double extraction_seconds = 0.0;
};

float degreesToRadians(const float degrees)
{
    constexpr float pi = 3.14159265358979323846f;
    return degrees * pi / 180.0f;
}

std::string toLower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

bool loadCloudAny(const std::string& filename, PointCloudT& cloud)
{
    namespace fs = std::filesystem;

    const std::string extension = toLower(fs::path(filename).extension().string());

    std::cout << "Loading file: " << filename << '\n';
    std::cout << "Detected extension: " << extension << '\n';

    int status = -1;

    if (extension == ".pcd")
    {
        std::cout << "Using the PCD loader.\n";
        status = pcl::io::loadPCDFile<PointT>(filename, cloud);
    }
    else if (extension == ".ply")
    {
        std::cout << "Using the PLY loader.\n";
        status = pcl::io::loadPLYFile<PointT>(filename, cloud);
    }
    else
    {
        std::cerr << "Unsupported file type. Please use .pcd or .ply files.\n";
        return false;
    }

    if (status < 0)
    {
        std::cerr << "Failed to read file: " << filename << '\n';
        return false;
    }

    if (cloud.empty())
    {
        std::cerr << "The file was read, but the cloud is empty: " << filename << '\n';
        return false;
    }

    return true;
}

void printCloudSummary(const std::string& label, const PointCloudPtr& cloud)
{
    std::cout << label << " contains " << cloud->size() << " points.\n";
}

BoundingBox computeCombinedBoundingBox(const PointCloudT& source_cloud,
                                       const PointCloudT& target_cloud)
{
    BoundingBox bbox{};

    pcl::getMinMax3D(source_cloud, bbox.min_pt, bbox.max_pt);

    PointT target_min;
    PointT target_max;
    pcl::getMinMax3D(target_cloud, target_min, target_max);

    bbox.min_pt.x = std::min(bbox.min_pt.x, target_min.x);
    bbox.min_pt.y = std::min(bbox.min_pt.y, target_min.y);
    bbox.min_pt.z = std::min(bbox.min_pt.z, target_min.z);

    bbox.max_pt.x = std::max(bbox.max_pt.x, target_max.x);
    bbox.max_pt.y = std::max(bbox.max_pt.y, target_max.y);
    bbox.max_pt.z = std::max(bbox.max_pt.z, target_max.z);

    return bbox;
}

void printBoundingBox(const BoundingBox& bbox)
{
    std::cout << "Combined bounding box:\n"
              << "  Min: (" << bbox.min_pt.x << ", " << bbox.min_pt.y << ", "
              << bbox.min_pt.z << ")\n"
              << "  Max: (" << bbox.max_pt.x << ", " << bbox.max_pt.y << ", "
              << bbox.max_pt.z << ")\n";
}

Eigen::Vector4f computeCentroid(const PointCloudPtr& cloud, const std::string& label)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    std::cout << label << " centroid: (" << centroid[0] << ", " << centroid[1] << ", "
              << centroid[2] << ")\n";

    return centroid;
}

NormalCloudPtr estimateNormals(const PointCloudPtr& cloud,
                               const int k_search,
                               const std::string& label)
{
    pcl::NormalEstimation<PointT, NormalT> estimator;
    estimator.setInputCloud(cloud);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    estimator.setSearchMethod(tree);
    estimator.setKSearch(k_search);

    NormalCloudPtr normals(new NormalCloudT);
    estimator.compute(*normals);

    std::cout << "Computed normals for " << label << ".\n";
    return normals;
}

void flipNormalsTowardCentroid(const PointCloudPtr& cloud,
                               const NormalCloudPtr& normals,
                               const Eigen::Vector4f& centroid)
{
    for (std::size_t i = 0; i < cloud->size(); ++i)
    {
        Eigen::Vector3f vector_to_centroid = centroid.head<3>() - cloud->points[i].getVector3fMap();
        if (vector_to_centroid.norm() < 1e-6f)
        {
            continue;
        }
        vector_to_centroid.normalize();

        Eigen::Vector3f normal(normals->points[i].normal_x,
                               normals->points[i].normal_y,
                               normals->points[i].normal_z);
        if (normal.norm() < 1e-6f)
        {
            continue;
        }
        normal.normalize();

        // If the normal points away from the centroid, flip it.
        if (normal.dot(vector_to_centroid) < 0.0f)
        {
            normals->points[i].normal_x *= -1.0f;
            normals->points[i].normal_y *= -1.0f;
            normals->points[i].normal_z *= -1.0f;
        }
    }
}

GridDefinition createGrid(const BoundingBox& bbox, const double box_size)
{
    GridDefinition grid;
    grid.size_x = static_cast<float>(box_size);
    grid.size_y = static_cast<float>(box_size);
    grid.size_z = static_cast<float>(box_size);

    grid.divisions_x = std::max(1, static_cast<int>(std::ceil((bbox.max_pt.x - bbox.min_pt.x) / grid.size_x)));
    grid.divisions_y = std::max(1, static_cast<int>(std::ceil((bbox.max_pt.y - bbox.min_pt.y) / grid.size_y)));
    grid.divisions_z = std::max(1, static_cast<int>(std::ceil((bbox.max_pt.z - bbox.min_pt.z) / grid.size_z)));
    grid.total_boxes = grid.divisions_x * grid.divisions_y * grid.divisions_z;

    return grid;
}

void printGridSummary(const GridDefinition& grid)
{
    std::cout << "Voxel grid summary:\n"
              << "  Size: (" << grid.size_x << ", " << grid.size_y << ", " << grid.size_z
              << ") meters\n"
              << "  Divisions: " << grid.divisions_x << " x " << grid.divisions_y << " x "
              << grid.divisions_z << '\n'
              << "  Total boxes: " << grid.total_boxes << '\n';
}

int getBoxIndex(const PointT& point, const BoundingBox& bbox, const GridDefinition& grid)
{
    const int i = static_cast<int>((point.x - bbox.min_pt.x) / grid.size_x);
    const int j = static_cast<int>((point.y - bbox.min_pt.y) / grid.size_y);
    const int k = static_cast<int>((point.z - bbox.min_pt.z) / grid.size_z);

    if (i < 0 || i >= grid.divisions_x)
    {
        return -1;
    }
    if (j < 0 || j >= grid.divisions_y)
    {
        return -1;
    }
    if (k < 0 || k >= grid.divisions_z)
    {
        return -1;
    }

    return i * grid.divisions_y * grid.divisions_z + j * grid.divisions_z + k;
}

BoxCollections createEmptyBoxCollections(const int total_boxes)
{
    BoxCollections boxes;

    boxes.source_boxes.resize(total_boxes);
    boxes.target_boxes.resize(total_boxes);
    boxes.source_normal_boxes.resize(total_boxes);
    boxes.target_normal_boxes.resize(total_boxes);

    for (int index = 0; index < total_boxes; ++index)
    {
        boxes.source_boxes[index].reset(new PointCloudT);
        boxes.target_boxes[index].reset(new PointCloudT);
        boxes.source_normal_boxes[index].reset(new NormalCloudT);
        boxes.target_normal_boxes[index].reset(new NormalCloudT);
    }

    return boxes;
}

void distributeCloudIntoBoxes(const PointCloudPtr& cloud,
                              const NormalCloudPtr& normals,
                              const BoundingBox& bbox,
                              const GridDefinition& grid,
                              std::vector<PointCloudPtr>& point_boxes,
                              std::vector<NormalCloudPtr>& normal_boxes)
{
    for (std::size_t point_index = 0; point_index < cloud->size(); ++point_index)
    {
        const PointT& point = cloud->points[point_index];
        const int box_index = getBoxIndex(point, bbox, grid);

        if (box_index < 0)
        {
            continue;
        }

        point_boxes[box_index]->points.push_back(point);
        normal_boxes[box_index]->points.push_back(normals->points[point_index]);
    }
}

PlanarSelectionResult extractPlanarSubsets(const BoxCollections& boxes,
                                           const Parameters& params)
{
    PlanarSelectionResult result;

    const auto start_time = std::chrono::steady_clock::now();
    const float cos_angle_threshold = std::cos(degreesToRadians(params.angle_threshold_degrees));

    for (std::size_t index = 0; index < boxes.source_boxes.size(); ++index)
    {
        const PointCloudPtr& source_box = boxes.source_boxes[index];
        const PointCloudPtr& target_box = boxes.target_boxes[index];
        const NormalCloudPtr& source_normals = boxes.source_normal_boxes[index];
        const NormalCloudPtr& target_normals = boxes.target_normal_boxes[index];

        // Skip boxes that are empty in both clouds.
        if (source_box->empty() && target_box->empty())
        {
            continue;
        }

        // A box is only evaluated if both clouds contain enough points.
        if (source_box->size() < params.min_points_per_box ||
            target_box->size() < params.min_points_per_box)
        {
            continue;
        }

        std::vector<Eigen::Vector3f> all_normals;
        all_normals.reserve(source_normals->size() + target_normals->size());

        // Add valid source normals.
        for (const NormalT& normal : source_normals->points)
        {
            Eigen::Vector3f vector(normal.normal_x, normal.normal_y, normal.normal_z);
            if (vector.norm() > 1e-6f)
            {
                all_normals.push_back(vector.normalized());
            }
        }

        // Add valid target normals.
        for (const NormalT& normal : target_normals->points)
        {
            Eigen::Vector3f vector(normal.normal_x, normal.normal_y, normal.normal_z);
            if (vector.norm() > 1e-6f)
            {
                all_normals.push_back(vector.normalized());
            }
        }

        if (all_normals.empty())
        {
            continue;
        }

        // Compute the average normal direction of all normals inside this box.
        Eigen::Vector3f average_normal(0.0f, 0.0f, 0.0f);
        for (const Eigen::Vector3f& normal : all_normals)
        {
            average_normal += normal;
        }

        if (average_normal.norm() < 1e-6f)
        {
            continue;
        }
        average_normal.normalize();

        // Count how many normals are aligned with the average direction.
        std::size_t consistent_normals = 0;
        for (const Eigen::Vector3f& normal : all_normals)
        {
            const float dot_product = average_normal.dot(normal);
            if (dot_product >= cos_angle_threshold)
            {
                ++consistent_normals;
            }
        }

        const float consistency_ratio =
            static_cast<float>(consistent_normals) / static_cast<float>(all_normals.size());

        if (consistency_ratio >= params.consistency_threshold)
        {
            // The box is sufficiently planar, so keep all points from this box.
            *result.planar_source += *source_box;
            *result.planar_target += *target_box;
        }
    }

    const auto end_time = std::chrono::steady_clock::now();
    result.extraction_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

bool saveCloudIfNotEmpty(const PointCloudPtr& cloud, const std::string& filename)
{
    if (cloud->empty())
    {
        return false;
    }

    const int status = pcl::io::savePCDFileASCII(filename, *cloud);
    if (status < 0)
    {
        std::cerr << "Failed to save cloud: " << filename << '\n';
        return false;
    }

    std::cout << "Saved cloud: " << filename << '\n';
    return true;
}

bool runGicp(const PointCloudPtr& source_cloud,
             const PointCloudPtr& target_cloud,
             const Parameters& params,
             Eigen::Matrix4f& final_transformation,
             double& fitness_score,
             double& runtime_seconds)
{
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(source_cloud);
    gicp.setInputTarget(target_cloud);
    gicp.setMaximumIterations(params.gicp_max_iterations);
    gicp.setMaxCorrespondenceDistance(params.gicp_max_correspondence_distance);
    gicp.setTransformationEpsilon(params.gicp_transformation_epsilon);
    gicp.setEuclideanFitnessEpsilon(params.gicp_euclidean_fitness_epsilon);
    gicp.setUseReciprocalCorrespondences(params.gicp_use_reciprocal_correspondences);

    PointCloudT aligned_source;

    const auto start_time = std::chrono::steady_clock::now();
    gicp.align(aligned_source);
    const auto end_time = std::chrono::steady_clock::now();

    runtime_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (!gicp.hasConverged())
    {
        return false;
    }

    final_transformation = gicp.getFinalTransformation();
    fitness_score = gicp.getFitnessScore();
    return true;
}

void printParameters(const Parameters& params)
{
    std::cout << "Parameters used by the program:\n"
              << "  Minimum points per box: " << params.min_points_per_box << '\n'
              << "  Planarity angle threshold: " << params.angle_threshold_degrees << " degrees\n"
              << "  Normal consistency threshold: " << params.consistency_threshold << '\n'
              << "  Normal estimation k-search: " << params.k_search << '\n'
              << "  GICP max iterations: " << params.gicp_max_iterations << '\n'
              << "  GICP max correspondence distance: "
              << params.gicp_max_correspondence_distance << " m\n"
              << "  GICP transformation epsilon: " << params.gicp_transformation_epsilon << '\n'
              << "  GICP euclidean fitness epsilon: "
              << params.gicp_euclidean_fitness_epsilon << '\n'
              << "  GICP reciprocal correspondences: "
              << (params.gicp_use_reciprocal_correspondences ? "true" : "false") << "\n\n";
}

void printUsage(const char* executable_name)
{
    std::cerr << "Usage: " << executable_name
              << " <source_cloud(.pcd|.ply)> <target_cloud(.pcd|.ply)> <box_size_in_meters>\n"
              << "Example: " << executable_name << " source.ply target.pcd 2.0\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::cout << "Build stamp: " << __DATE__ << ' ' << __TIME__ << "\n\n";

    if (argc < 4)
    {
        printUsage(argv[0]);
        return -1;
    }

    const std::string source_filename = argv[1];
    const std::string target_filename = argv[2];

    double box_size = 0.0;
    try
    {
        box_size = std::stod(argv[3]);
    }
    catch (const std::exception&)
    {
        std::cerr << "Error: box_size_in_meters must be a valid number.\n";
        return -1;
    }

    if (box_size <= 0.0)
    {
        std::cerr << "Error: box_size_in_meters must be positive.\n";
        return -1;
    }

    const Parameters params;
    printParameters(params);

    PointCloudPtr source_cloud(new PointCloudT);
    PointCloudPtr target_cloud(new PointCloudT);

    if (!loadCloudAny(source_filename, *source_cloud))
    {
        std::cerr << "Could not load the source cloud.\n";
        return -1;
    }
    if (!loadCloudAny(target_filename, *target_cloud))
    {
        std::cerr << "Could not load the target cloud.\n";
        return -1;
    }

    printCloudSummary("Source cloud", source_cloud);
    printCloudSummary("Target cloud", target_cloud);
    std::cout << '\n';

    // Step 1: Compute the shared bounding box of both clouds.
    const BoundingBox bbox = computeCombinedBoundingBox(*source_cloud, *target_cloud);
    printBoundingBox(bbox);
    std::cout << '\n';

    // Step 2: Compute centroids. They are later used to orient normals consistently.
    const Eigen::Vector4f source_centroid = computeCentroid(source_cloud, "Source cloud");
    const Eigen::Vector4f target_centroid = computeCentroid(target_cloud, "Target cloud");
    std::cout << '\n';

    // Step 3: Estimate normals for both clouds.
    NormalCloudPtr source_normals = estimateNormals(source_cloud, params.k_search, "the source cloud");
    NormalCloudPtr target_normals = estimateNormals(target_cloud, params.k_search, "the target cloud");

    // Step 4: Flip normals so that the orientation is more consistent.
    flipNormalsTowardCentroid(source_cloud, source_normals, source_centroid);
    flipNormalsTowardCentroid(target_cloud, target_normals, target_centroid);
    std::cout << "Normals were oriented toward the cloud centroids.\n\n";

    // Step 5: Build a voxel grid over the combined bounding box.
    const GridDefinition grid = createGrid(bbox, box_size);
    printGridSummary(grid);
    std::cout << '\n';

    // Step 6: Assign points and their normals to the corresponding voxel.
    BoxCollections boxes = createEmptyBoxCollections(grid.total_boxes);
    distributeCloudIntoBoxes(source_cloud,
                             source_normals,
                             bbox,
                             grid,
                             boxes.source_boxes,
                             boxes.source_normal_boxes);
    distributeCloudIntoBoxes(target_cloud,
                             target_normals,
                             bbox,
                             grid,
                             boxes.target_boxes,
                             boxes.target_normal_boxes);
    std::cout << "Distributed source and target points into voxel boxes.\n\n";

    // Step 7: Keep only the boxes that are planar enough in both clouds.
    const PlanarSelectionResult planar_result = extractPlanarSubsets(boxes, params);
    std::cout << "Planar extraction finished in " << planar_result.extraction_seconds << " seconds.\n";
    printCloudSummary("Planar source subset", planar_result.planar_source);
    printCloudSummary("Planar target subset", planar_result.planar_target);
    std::cout << '\n';

    if (planar_result.planar_source->empty() || planar_result.planar_target->empty())
    {
        std::cout << "No valid planar subsets were found in both clouds.\n";
        std::cout << "The program will stop before GICP.\n";
        return 0;
    }

    // Step 8: Save the reduced planar clouds.
    // Relative output paths are used here so the code is portable across machines.
    saveCloudIfNotEmpty(planar_result.planar_source, "planar_source_combined.pcd");
    saveCloudIfNotEmpty(planar_result.planar_target, "planar_target_combined.pcd");
    std::cout << '\n';

    // Step 9: Run GICP only on the selected planar subsets.
    std::cout << "Starting GICP registration on the planar subsets...\n";

    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    double fitness_score = 0.0;
    double registration_seconds = 0.0;

    const bool converged = runGicp(planar_result.planar_source,
                                   planar_result.planar_target,
                                   params,
                                   transformation,
                                   fitness_score,
                                   registration_seconds);

    if (!converged)
    {
        std::cout << "GICP did not converge.\n";
        return 0;
    }

    std::cout << "GICP converged.\n"
              << "Fitness score: " << fitness_score << '\n'
              << "Time taken for GICP registration: " << registration_seconds << " seconds\n"
              << "Transformation matrix:\n"
              << transformation << "\n\n";

    // Optional extra step:
    // If needed, the original source cloud can be transformed with the estimated matrix.
    PointCloudT transformed_source;
    pcl::transformPointCloud(*source_cloud, transformed_source, transformation);
    std::cout << "The original source cloud was also transformed in memory.\n";
    std::cout << "Finished processing planar selection and registration.\n";

    return 0;
}
