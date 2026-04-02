# Release notes

## v0.1

Initial public repository version of **SSC-Registration**.

### Included
- SSC-based adaptive fragmentation workflow
- PV-GICP-related registration components
- CMake-based build setup
- README documentation
- Citation file
- MIT license
- Git ignore for build and temporary files

### SSC features
- input point cloud loading from `.pcd` and `.ply`
- automatic GPS time field detection
- initial GPS-time-based splitting
- adaptive fragment growth
- whole-fragment normal recomputation
- semi-sphere seed clustering
- fragment export and summary output
- optional maximum-extent acceptance rule


**Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation**

Some implementation details are configurable and may include project-specific engineering options beyond the minimal method description in the paper.

### Known limitations
- large point clouds may require careful memory handling
- viewer support depends on local PCL / VTK build configuration
- some options are tuned for practical workflow use rather than strict article reproduction
