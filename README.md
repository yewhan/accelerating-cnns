# Accelerating CNNs: Powering Up Deep Learning Performance
Optimising the Performance of Real World Machine Learning Algorithms on Modern Multi-Core Processors

The code is split up into both a CPU-enabled Convolution-ReLU layer, and a GPU-enabled Convolution-ReLU layer.

## CPU
The CPU portion of the code was designed for a Ryzen 5 3600 CPU.
Multiple versions of the code was developed, each utilising a slightly different set of/ order of optimisations applied.

## GPU
The GPU portion of the code was designed for a GTX 1080 Ti.

## Optimisations Applied
### CPU
The CPU-enabled code leverages the following optimisations:

- Vectorisation via SIMD,
- Parallelisation via OpenMP,
- Quantisation,
- Loop Unrolling,
- Strength Reduction,
- Moving Code Outside of Loops,
- Common Subscript Elimination,
- Loop Interchange,
- Replacing Function Calls with Inlined Functions,
- Constant Declaration,
- Declaring Variables Inside of the Loop Body,
- Loop Tiling.

### GPU
The GPU-enabled code leverages the following optimisations:

- Asynchronous Memory Transfers,
- Parallelisation via SIMT

## Running the Code
### CPU
The CPU-enabled code was compiled on Linux using the Clang compiler. Due to issues other compilers (GCC) faced when attempting to collapse loops, it may be necessary to compile with Clang.
All code was compiled using the following command: **Clang -O3 -fopenmp -march=native -mavx2 -lm -Rpass=loop-vectorise -pthread -D_GNU_SOURCE -g -xc {source_file_location} -o o**

### GPU
The GPU-enabled code was compiled directly inside of Visual Studio 2022, using the "Release" build.
