# SSC-Registration

Code and data for the paper:

**Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation**

---

## Related article

**Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation**  
Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst

---

## Workflow overview

<img width="2903" height="1209" alt="Workflow5" src="https://github.com/user-attachments/assets/be19e596-510b-4611-990e-c6f528ef7d92" />

---

## Abstract

This study presents a novel workflow designed to efficiently and accurately register large-scale mobile laser scanning (MLS) point clouds to a target model point cloud in urban street scenarios. The workflow specifically targets the complexities inherent in urban environments and addresses the challenges of integrating point clouds that vary in density, noise characteristics, and occlusion scenarios, which are common in busy city centers.

Two methodological advancements are introduced. First, the proposed **Semi-sphere Check (SSC)** preprocessing technique adaptively fragments MLS trajectory data by identifying mutually orthogonal planar surfaces. This step reduces the impact of MLS drift on the accuracy of the entire point cloud registration, while ensuring sufficient geometric features within each fragment to avoid local minima.

Second, the workflow introduces **Planar Voxel-based Generalized Iterative Closest Point (PV-GICP)**, a fine registration method that selectively utilizes planar surfaces within voxel partitions. This strategy improves registration accuracy and reduces computation time by more than 50% compared to conventional point-to-plane ICP methods.

Experiments on real-world datasets from Munich’s inner city demonstrate that the workflow achieves sub-0.01 m average registration accuracy while significantly shortening processing times. The results highlight the potential of the proposed methods for automated 3D urban modeling and updating, with direct applications in urban planning, infrastructure management, and dynamic city monitoring.

---

## Overview

This repository contains a practical implementation of two main stages of the workflow presented in the paper:

- **Semi-sphere Check (SSC)**  
  Adaptive fragmentation of MLS point clouds to reduce drift effects while preserving enough geometric structure for stable registration.

- **Planar Voxel-based Generalized ICP (PV-GICP)**  
  Fine registration using planar voxel subsets instead of the full point cloud.

The overall goal is to divide long MLS trajectories into fragments that are:

- small enough to reduce the effect of drift
- large enough to contain reliable planar structure for registration

---

## Repository structure

```text
include/    Public headers and parameter definitions
src/        Source code
build/      Local build directory
data/       Optional input or example data
