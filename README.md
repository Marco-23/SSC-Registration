# SSC-Registration
Code and data for "Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation"



# Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation

This repository contains the implementation code, data samples, and documentation for the paper:

**Ortiz Rincón, M.A., Yang, Y., Holst, C. (2025). Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation.**

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

