# ParticleFX: High-Performance Particle Rendering with OpenMP and CUDA

## Project Overview

ParticleFX is a high-performance particle rendering system that implements and analyzes rendering techniques using both OpenMP and CUDA. The system enhances image contrast through a three-stage process involving localized histogram creation, histogram equalization, and interpolation. This project compares the performance of CPU-based OpenMP implementation with GPU-accelerated CUDA implementation, showcasing the efficiency gains achieved through parallel computing.

## Project Structure
- **`config.h`**: Configuration settings and constants used throughout the project.
- **`common.h`/`helper.h`**: Common definitions and utility functions used across different implementations.
- **`cpu.h`/`openmp.h`/`cuda.cuh`**: Implements CPU/OpenMP/CUDA-based particle rendering, including kernels for various stages of processing respectively. 
- **`cpu.c`**: Original CPU-based implementation. Provides single-threaded functionality for particle rendering.
- **`openmp.c`**: OpenMP-based implementation. Extends the CPU implementation with parallelization for performance improvement.
- **`main.cu`**: Entry point for the CUDA implementation. Contains the CUDA kernel launches and manages data transfer between the host and device.
- **`cuda.cu`**: contains CUDA functions for initializing resources, managing memory, and executing kernel operations for particle rendering. It includes implementations for handling pixel contributions, sorting data, and blending colors to produce the final image.
- **`Makefile`**: Build instructions for compiling the project, including targets for CPU, OpenMP, and CUDA builds.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Vyshnav12/ParticleFX.git
    cd ParticleFX
    ```

2. Build the project:

    Ensure you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [STB Image Write](https://github.com/nothings/stb) installed. Use `make` to compile:

    ```bash
    make
    ```

## Usage

Run the particle rendering application with the following command:

```bash
./particle_render <mode> <particle_count> <output_image_dimensions> [<output_image>] [--bench]
```
### Arguments

- `<mode>`: Rendering mode (CPU, OPENMP, or CUDA)
- `<particle_count>`: Number of particles to render
- `<output_image_dimensions>`: Dimensions of the output image (e.g., 512 or 512x1024)
- `<output_image>` (optional): Path to save the output image (must end with .png)
- `--bench` (optional): Enable benchmarking mode to run multiple iterations and gather performance data

### Example

```bash
./particle_render CUDA 10000 512x512 output.png --bench
```

This command runs the CUDA implementation with 10,000 particles, saves the output as output.png, and enables benchmarking.

## Original CPU Implementation

The original particle rendering is implemented in `cpu.c`, which serves as the baseline for performance comparison. This implementation handles the entire rendering process on the CPU without parallelization. It includes:

- **Initialization of Particle Data**: Setting up initial conditions and properties for the particles.
- **Pixel Contribution Calculation**: Computing how each particle contributes to the image based on its properties.
- **Pair Sorting**: Organizing particle data to facilitate efficient blending.
- **Blending**: Combining pixel contributions from all particles to produce the final image.
- **Resource Management**: Releasing resources and cleaning up after processing.

This single-threaded approach provides a reference point to evaluate the performance improvements achieved by parallelizing the rendering process using OpenMP and CUDA.

## Implementation Details

### OpenMP Implementation

The OpenMP implementation leverages multi-threading to accelerate the particle rendering process on the CPU. The project includes functions for:

- **Initializing Particle Data**
- **Computing Pixel Contributions**
- **Sorting Pairs**
- **Blending**
- **Releasing Resources**

**Stages Implemented:**
- **Stage 1:** Localized histogram creation.
- **Stage 2:** Histogram equalization.
- **Stage 3:** Interpolation to enhance image contrast.

### CUDA Implementation

The CUDA implementation is designed to utilize the GPU's parallel computing power, further enhancing performance. The CUDA kernels perform similar tasks as the OpenMP stages but are optimized for GPU execution.

**Stages Implemented:**
- **Stage 1:** Histogram creation optimized with techniques like loop unrolling and grid/block size optimization.
- **Stage 2:** Histogram equalization with potential for further optimization using parallel merge sort and grid/block optimization.
- **Stage 3:** Contrast interpolation, highly efficient.


## Performance Results

**OpenMP Results**

| Particle Count | Image Size | CPU Reference Timing (ms) | OpenMP Stage 1 Timing (ms) | OpenMP Stage 2 Timing (ms) | OpenMP Stage 3 Timing (ms) |
|----------------|------------|---------------------------|-----------------------------|-----------------------------|-----------------------------|
| 10,000         | 1000x1000   | 14.860                    | 7.806                       | 63.009                      | 3.584                       |
| 50,000         | 2000x2000   | 105.438                   | 30.126                      | 335.437                     | 15.693                      |
| 100,000        | 3000x3000   | 227.544                   | 72.821                      | 689.283                     | 31.394                      |
| 500,000        | 3800x3800   | 1029.060                  | 350.066                     | 3807.812                    | 139.959                     |
| 1,000,000      | 4000x4000   | 2352.572                  | 622.965                     | 7522.185                    | 232.384                     |

**CUDA Results**

| Particle Count | Image Size | CPU Reference Timing (ms) | CUDA Stage 1 Timing (ms) | CUDA Stage 2 Timing (ms) | CUDA Stage 3 Timing (ms) |
|----------------|------------|---------------------------|--------------------------|--------------------------|--------------------------|
| 10,000         | 1000x1000   | 14.860                    | 0.0826                   | 11.668                   | 0.194                    |
| 50,000         | 2000x2000   | 105.438                   | 8.814                    | 70.008                   | 0.829                    |
| 100,000        | 3000x3000   | 227.544                   | 21.474                   | 146.526                  | 2.133                    |
| 500,000        | 3800x3800   | 1029.060                  | 122.029                  | 718.588                  | 9.278                    |
| 1,000,000      | 4000x4000   | 2352.572                  | 228.510                  | 1513.318                 | 26.898                   |

**Hardware**

The performance results were obtained on the following hardware:

- **GPU**: NVIDIA RTX 3060 (Mobile)
- **CPU**: Intel Core i5-11400H
- **RAM**: 16 GB
- **Operating System**: Windows 11

## Profiling Insights for CUDA

- **Stage 1:** Latency-bound, with opportunities for loop unrolling and grid/block size optimization.
- **Stage 2:** Latency-bound with balanced throughput; potential improvements include parallel merge sort and further grid/block optimization.
- **Stage 3:** Memory-bound with an L2 Cache bottleneck; additional optimization techniques from Stage 1 can be applied.

## Validation

Output images are validated by comparing them against a reference generated using the CPU implementation. The validation checks include:

- **Image Dimensions**: Ensure the output image matches the specified dimensions.
- **Pixel Accuracy**: Verify that pixel colors are within an acceptable range of the reference image.

## Known Issues

- **CUDA Compatibility**: Ensure the correct version of the CUDA Toolkit is installed. The project may not run on systems without compatible NVIDIA GPUs.
- **Performance Variations**: Timing results may vary based on system specifications and hardware configurations.

## Development Environment

- **IDE:** [Visual Studio](https://visualstudio.microsoft.com/)
- **Profiling Tools:** [NSight Profiler](https://developer.nvidia.com/nsight-systems) for CUDA kernels

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
