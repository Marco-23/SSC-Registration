# SSC-Registration
Code and data for "Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation"



# Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation

<img width="2903" height="1209" alt="Workflow5" src="https://github.com/user-attachments/assets/be19e596-510b-4611-990e-c6f528ef7d92" />


# Abstract
This study presents a novel workflow designed to efficiently and accurately register large-scale mobile laser scanning (MLS) point clouds to a target model point cloud in urban street scenarios. This workflow specifically targets the complexities inherent in urban environments and adeptly addresses the challenges of integrating point clouds that vary in density, noise characteristics, and occlusion scenarios, which are common in bustling city centers. Two methodological advancements are introduced. First, the proposed Semi-sphere Check (SSC) preprocessing technique optimally fragments MLS trajectory data by identifying mutually orthogonal planar surfaces. This step reduces the impact of MLS drift on the accuracy of the entire point cloud registration, while ensuring sufficient geometric features within each fragment to avoid local minima.
Second, we propose Planar Voxel-based Generalized Iterative Closest Point (PV-GICP), a fine registration method that selectively utilizes planar surfaces within voxel partitions. This pre-process strategy not only improves registration accuracy but also reduces computation time by more than 50\% compared to conventional point-to-plane ICP methods.
Experiments on real-world datasets from Munich’s inner city demonstrate that our workflow achieves sub-0.01 m average registration accuracy while significantly shortening processing times. The results underscore the potential of the proposed methods to advance automated 3D urban modeling and updating, with direct applications in urban planning, infrastructure management, and dynamic city monitoring.

## 📖 Overview
This workflow introduces:
- **Semi-Sphere Check (SSC)**: Adaptive fragmentation of MLS data to reduce drift effects.
- **Planar Voxel-based Generalized ICP (PV-GICP)**: Efficient fine registration using planar voxels.

## 🧰 Requirements
```bash
CMake ≥ 3.20
PCL ≥ 1.12
Eigen3
PDAL  optional

