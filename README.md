# SSC-Registration

Code and data for the paper:

**Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation**

---

## Related article

**Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation**  
Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst  
*International Journal of Applied Earth Observation and Geoinformation*  
149 (2026) 105272  
DOI: 10.1016/j.jag.2026.105272

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


## Software overview

<img width="3832" height="2035" alt="SSC" src="https://github.com/user-attachments/assets/f8507213-2806-4721-b77e-626805f4d26f" />

---

## Software Download

The latest Windows version is available on the [Releases](../../releases/latest) page.

### Windows
1. Download `SSC-1.0.0-win64.zip`
2. Extract the ZIP file
3. Run `SSC-1.0.0-win64.exe`

---

## How to use this repository

This repository is organized into two main parts:

- **`SSC/`**  
  Contains the adaptive fragmentation workflow based on the Semi-sphere Check.

- **`PV-GICP/`**  
  Contains the planar-voxel-based fine registration workflow.

If you want to work on **adaptive fragmentation**, start in `SSC/`.  
If you want to work on **fine registration**, start in `PV-GICP/`.

---

## Repository structure

```text
SSC/        Semi-sphere Check adaptive fragmentation workflow
PV-GICP/    Planar Voxel-based GICP workflow
include/    Public headers and parameter definitions
src/        Source code

```

---

## Requirements

```text
CMake >= 3.20
PCL >= 1.12
Eigen3
PDAL (optional, depending on build configuration)
VTK / PCL visualization modules (optional, for viewer support)
```

---

## Build

### Windows

```bash
cmake -S . -B build
cmake --build build --config Release
```

### Linux

```bash
cmake -S . -B build
cmake --build build -j
```

---

## Quick start

A typical SSC run looks like this:

```bash
./build/Release/ssc_full_pipeline.exe "path/to/input_cloud.pcd" "path/to/output_dir"
```

Example with useful runtime options:

```bash
./build/Release/ssc_full_pipeline.exe "path/to/input_cloud.pcd" "path/to/output_dir" --no-save-initial-frames --no-preview-ply --accept-if-extent-over 205
```

Example with viewer enabled:

```bash
./build/Release/ssc_full_pipeline.exe "path/to/input_cloud.pcd" "path/to/output_dir" --show
```

---

## SSC command line usage

```text
ssc_full_pipeline <input_cloud(.pcd|.ply)> <output_directory> [options]
```

### Main options

| Flag | Meaning |
|---|---|
| `--split-seconds <value>` | Initial GPS-time split duration |
| `--normal-k <value>` | Number of neighbors used for normal estimation |
| `--min-fragment-extent <meters>` | Minimum candidate extent before SSC acceptance is allowed |
| `--max-extra-frames <value>` | Maximum number of additional initial frames that may be merged |
| `--min-populated-seeds <value>` | Minimum number of sufficiently populated seed clusters |
| `--min-seed-points <value>` | Minimum number of normals required for a seed to count as populated |
| `--max-mean-disp <value>` | Mean seed displacement threshold |
| `--max-std-disp <value>` | Standard deviation threshold of seed displacement |
| `--accept-if-extent-over <meters>` | Project-specific rule: automatically accept a fragment above this extent |
| `--no-save-initial-frames` | Do not write the initial GPS split frames |
| `--no-preview-ply` | Do not write the colored preview PLY |
| `--show` | Request visualization if viewer support is available |

---

## Input data requirements

The SSC pipeline expects an input point cloud in `.pcd` or `.ply` format.

### Required fields

- `x`
- `y`
- `z`

### Required GPS time field

A GPS-time-like scalar field must be available for the initial splitting step. Common accepted names include:

- `Gps_Time`
- `gps_time`
- `scalar_Gps_Time`

The field is detected automatically.

### Optional fields

Optional attributes such as intensity, return number, number of returns, range, or RGB can also be present.

---


## Citation

If you use this repository in research, please cite the paper.

```bibtex
@article{OrtizRincon2026SSC,
  title   = {Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation},
  author  = {Ortiz Rincón, Marco Antonio and Yang, Yihui and Holst, Christoph},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  volume  = {149},
  pages   = {105272},
  year    = {2026},
  doi     = {10.1016/j.jag.2026.105272}
}
```
---



## License

The source code in this repository is licensed under the **MIT License**.

The related article and any manuscript-related materials included in this repository remain subject to their own publication license. The preprint version of the article is available under **CC BY-NC-ND 4.0**.

---

## Contact

Marco Antonio Ortiz Rincón  
GitHub: [Marco-23](https://github.com/Marco-23)
