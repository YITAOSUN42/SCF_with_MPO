#  Self-consistent tensor network method for correlated super-moiré matter beyond one billion sites

[![arXiv](https://img.shields.io/badge/arXiv-2302.04919-B31B1B)](https://arxiv.org/abs/2503.04373)

This repository is for the manuscript "Self-consistent tensor network method for correlated super-moiré matter beyond one billion sites" by Yitao Sun, Marcel Niedermeier, Tiago V. C. Antão, Adolfo O. Fumega and Jose L. Lado. The code is still under heavy development without detailed optimization or refactorization.This repo contains:

* All data generated for the spectral functions shown in the manuscript.
* The "MPO_SCF" .jl files and relating notebook files, 2 for spatially varying on-site interaction, 2 for spatially varying hopping amplitude.
* Notebooks for calculation of spectral functions using stochastic tracing, inclduing "random_vec_generator" and "Cal_LDOS", "Plot_LDOS" for plotting.
* A get_function.jl with different functions used in notebooks above.

Codes are fully based on julia. To use the codes, you need to install these particular packages:

* ITensors.jl
* Quantics.jl
* TensorCrossInterpolation.jl
* QuanticsTCI.jl
