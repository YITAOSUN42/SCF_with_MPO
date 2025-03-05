#  Solving correlated super-moir´e states above a billion sites with a tensor network self-consistent method

This repository contains:

* All data generated for the spectral functions shown in the manuscript.
* The "MPO_SCF" .jl files and relating notebook files, 2 for spatially varying on-site interaction, 2 for spatially varying hopping amplitude.
* Notebooks for calculation of spectral functions using stochastic tracing, inclduing "random_vec_generator" and "Cal_LDOS", "Plot_LDOS" for plotting.
* A get_function.jl with different functions used in notebooks above.

Codes are fully based on julia. To use the codes, you need to install these particular packages:

* ITensors.jl
* Quantics.jl
* TensorCrossInterpolation.jl
* QuanticsTCI.jl
