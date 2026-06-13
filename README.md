# Blocked Matrix Multiplication for Alveo U50 with Multiple CUs

This repository contains the source codes, hardware design files, and benchmarks required to reproduce the proposed architecture from the paper "Optimizing Matrix Multiplication on FPGAs using Spatial Parallelism and High-Level Synthesis".

The project investigates the optimization of blocked matrix multiplication on the AMD Alveo U50 FPGA using High-Level Synthesis (HLS). The proposed hardware accelerator leverages spatial parallelism by instantiating up to four independent Compute Units (CUs), mapped to separate High Bandwidth Memory (HBM) banks, and evaluates their performance and energy efficiency against a multi-threaded CPU implementation.

## Repository Structure

* `krnl_mult.cpp`: The HLS kernel source code implementing the blocked matrix multiplication. It uses directives such as `DATAFLOW`, `ARRAY_PARTITION`, `PIPELINE`, and `UNROLL` to process 128x128 data blocks efficiently.
* `connectivity.cfg`: The configuration file that defines the instantiation of the 4 CUs and maps their AXI interfaces to specific HBM banks.
* `mmult_ref_openmp_1.c`, `mmult_ref_openmp_2.c`, `mmult_ref_openmp_4.c`: Multi-threaded CPU reference implementations using OpenMP (1, 2, and 4 threads) for performance comparison.
* `Benchmark.ipynb`: A Jupyter Notebook utilizing the PyXRT API to orchestrate the benchmark. It handles the dynamic compilation of CPU references, FPGA buffer allocation, execution of the CUs, functional verification, and energy consumption measurement via RAPL.

## Prerequisites

To compile and run this project, you will need:
* AMD Vitis and Vivado (tested with version 2024.1).
* Xilinx Runtime (XRT).
* Alveo U50 Data Center Accelerator Card and its deployment platform (`xilinx_u50_gen3x16_xdma_5_202210_1`).
* Python 3 with the following libraries: `numpy`, `pyxrt`, `pandas`.
* GCC with OpenMP support.

## Hardware Compilation

To reproduce the architecture, you need to compile the hardware kernel into an Xilinx Object (`.xo`) file and then link it to create the final executable binary (`.xclbin`). 

Use the following commands to compile the design for the Alveo U50 platform:

1.  Compile the HLS kernel:
    ```bash
    v++ -c -t hw --platform xilinx_u50_gen3x16_xdma_5_202210_1 -k krnl_mult krnl_mult.cpp -o krnl_mult.xo
    ```

2.  Link the kernel and apply the connectivity configuration for multiple CUs:
    ```bash
    v++ -l -t hw --platform xilinx_u50_gen3x16_xdma_5_202210_1 -s --config connectivity.cfg krnl_mult.xo -o krnl_mult.xclbin
    ```

## Running the Benchmark

Once the `krnl_mult.xclbin` file is generated, you can run the benchmarking suite provided in the Jupyter Notebook:

1.  Open `Benchmark.ipynb` in your Jupyter environment.
2.  Run all cells sequentially.
3.  The notebook will automatically:
    * Compile the CPU reference C codes into shared libraries (`.so`).
    * Initialize the FPGA and allocate buffers in the appropriate HBM banks.
    * Run matrix multiplication benchmarks for varying matrix sizes (from 4x4 up to 4096x4096).
    * Validate the functional correctness of the FPGA output against the CPU reference.
    * Measure the execution time and calculate the speedup.
    * Measure the energy consumption of the CPU using Intel RAPL.

## Authors
* Henrique Gregory Gimenez
* Edson Toshimi Midorikawa
* Felipe Valencia de Almeida
* Liria Matsumoto Sato

Departamento de Engenharia de Computação e Sistemas Digitais da Escola Politécnica - Universidade de São Paulo (USP)
