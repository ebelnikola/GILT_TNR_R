# GILT-TNR with rotations
This repository contains the Python3 and Julia codes for the numerical computations in the following papers:
- [Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group](https://arxiv.org/abs/2408.10312) 
- [Transfer Matrix and Lattice Dilatation Operator for High-Quality Fixed Points in Tensor Network Renormalization Group]()

Apart from computational packages for Python3 and Julia, our code relies heavily on the following libraries: [GiltTNR](https://github.com/GiltTNR/GiltTNR), [ncon](https://github.com/mhauru/ncon), and [abeliantensors](https://github.com/mhauru/abeliantensors).

All the source code is licensed under the MIT license, as described in the file LICENSE.

## Installation

1. Install Python3 with NumPy and SciPy packages. An easy way to do this is to use [anaconda](https://www.anaconda.com/download/). 
2. Install [Julia](https://julialang.org/). 
3. Clone this repository by running:
``` 
git clone https://github.com/ebelnikola/GILT_TNR_R 
```
4. Move to the repository:
```
cd GILT_TNR_R
```
5. Run the script that will install all necessary Julia packages:
```
julia install_packages.jl
```
If there are any problems at this step, try using the second version of the installation script as follows:
```
julia install_packages_v2.jl
``` 
The list of all required packages is provided in the `Manifest.toml` file. Note that `Manifest.toml` contains an exhaustive list of all dependencies (including the dependencies of dependencies). The packages used in the code explicitly are listed in `install_packages_v2.jl`.   

6. To run our interactive notebooks, you may want to install [jupyter notebook](https://jupyter.org/) or the corresponding [extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).     

## Notebooks

Once the installation is done, we invite the user to try our Jupyter notebooks, which allow one to reproduce some of our results easily. To run a notebook, open it either in VS Code (provided that the suitable extension is installed) or using the following command in the terminal (from the root of the repository):
```
jupyter notebook name_of_the_notebook
```

Notebooks:

`Newton_method_paper_results_reproduction.ipynb` - reproduces some of the results from [Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group](https://arxiv.org/abs/2408.10312). 

`Lattice_Dilatation_Operator_paper_results_reproduction.ipynb` - reproduces some of the results from [Transfer Matrix and Lattice Dilatation Operator for High-Quality Fixed Points in Tensor Network Renormalization Group]().

## Scripts 

We provide the following scripts in the Lab directory:

- `plot_trajectory.jl` applies the GiltTNR algorithm `traj_len` times to the initial tensor corresponding to 2d nearest neighbors Ising model at the relative temperature `relT` and anisotropy parameter `Jratio` ($a$ in [the Newton method paper](https://arxiv.org/abs/2408.10312)). It saves the resulting trajectories to the trajectories folder in two files: `*.data` with tensors, log factors, and errors along the trajectory; `*.log` with all the text output of the algorithm. Saves the plots of the trajectories of the tensor's singular values (obtained by "diagonal" SVD: $A_{ijkl}=U_{ij l} S_l V_{l kl}$).

- `critical_temperature.jl` finds the critical temperature using bisection search; saves the result into the "critical_temperatures" directory; plots the corresponding trajectories of the singular values. Saves the plot into the trajectory_plots directory. 
 
- `differentiability_test.jl` performs differentiability tests of the GiltTNR algorithm using a random direction vector. Saves the corresponding plot to the diff_tests folder.

- `eigensystem.jl` gets the largest eigenvalues and the corresponding eigenvectors of GiltTNR linearised around some initial approximation of the critical tensor (given by `relT`, `Jratio`, and `number_of_initial_steps`). Saves the resulting tensor and the eigensystem to the eigensystems folder. Note that the script will fix `bond_repetitions` and `recursion_depth`. These parameters will be saved together with the other output.  

- `newton.jl` (assumes that `rotation=true`) repeats the computation from `eigensystem.jl`. Then, it finds the critical tensor using Newton's method. Saves the data to the newton directory. Note that the found critical tensor is the fixed point for GiltTNR with fixed `bond_repetitions` and `recursion_depth`. These parameters will be saved together with the other data.

- `LDO_spectra.jl` computes the spectrum of the GILT-TNR operator. Saves the result in LDO folder.

Note that each script has the corresponding help describing all the command line arguments. To see this help run:
```
julia --project Lab/script_name.jl --help
```
### Naming convention

The database handling in this repository is not perfect. We acknowledge it could be improved, but we are following the saying "If it ain't broke, don't fix it". Important data generated by the code will use the following naming convention:
```
rotate=[rotate][chi][gilt_eps]_[cg_eps]__[info about particular experiment].[extension]
```
The first part of the name `rotate=[rotate]_[chi]_[gilt_eps]_[cg_eps]` uniquely characterizes the GiltTNR algorithm, with one caveat. This naming convention assumes that `bond_repetitions`, `recursion_depth`, and `Rmatrices` were not provided (see the list of adjustments in the GiltTNR directory section). Two functions generate database entries using this convention: `trajectory` and `plot_the_trajectory` (specifically, their methods with the `initialA_pars` argument). **To avoid ambiguities in the database, users should not pass `bond_repetitions`, `recursion_depth`, or `Rmatrices` to these functions.**

## Other files

### GiltTNR directory
This directory contains Python code implementing the GiltTNR algorithm (see [this paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111)). It combines code from [GiltTNR](https://github.com/GiltTNR/GiltTNR), [ncon](https://github.com/mhauru/ncon), and [abeliantensors](https://github.com/mhauru/abeliantensors). We made the following adjustments to the codes there:

1. Added an option to fix the number of Gilt algorithm iterations applied to each bond around a plaquette, controlled by the `bond_repetitions` keyword argument.

2. Added an option to fix the recursion depth in the optimization procedure for `R` matrices (`Q` matrices in the [Newton method paper](https://arxiv.org/abs/2408.10312)). This is controlled by the `recursion_depth` argument (used with `bond_repetitions`), which is a dictionary with keys from `{"S", "N", "E", "W"}` and values specifying recursion depths for corresponding legs. 

3. Added an option to use precomputed `R` matrices instead of running the optimization procedure. To use this, pass `bond_repetitions` and the `Rmatrices`, a dictionary with keys `(bond_key, lap)` (`lap` is an integer, `bond_key in {"S", "N", "E", "W"}`) and values as `R` matrices to be applied at the specified bond and Gilt iteration.

4. Added an option to control tensor rotation after the GiltTNR step, controlled by the `rotate` keyword.

5. Modified `matrix_eig` and `matrix_svd` functions in `abeliantensor.py` and `tensor.py` to remove sign ambiguities in the decomposition. Two methods of fixing the signs are available, controlled by the global variable `method` at the beginning of `abeliantensor.py` and `tensor.py`.

### Root directory

- `EchelonForm.jl` - technical code used in `GaugeFixing.jl` for checking ranks of boolean matrices. 
- `GaugeFixing.jl` - continuous and discrete gauge fixing routines.
- `install_packages.jl` - script that installs all the necessary dependencies.
- `KrylovTechnical.jl` - technical code that provides a minimal implementation of Z2 invariant tensors in Julia. This solves the problem of `KrylovKit` throwing a segmentation fault while working with PyObjects.
- `NumDifferentiation.jl` - provides the function that performs numerical differentiation.
- `Tools.jl` - Contains some useful functions. These are listed at the top of the file.     
- `IsingExactLevels` and `IsingEvenExactLevels` - data about exact 2d Ising spectrum.
- `Project.toml` and `Manifest.toml` - contain information about versions of packages used in the computations.
- `Newton_method_paper_results_reproduction.ipynb` - provides the simplest way to reproduce some of the results from [Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group](https://arxiv.org/abs/2408.10312).
- `Lattice_Dilatation_Operator_paper_results_reproduction.ipynb` - provides the simplest way to reproduce some of the results from [Transfer Matrix and Lattice Dilatation Operator for High-Quality Fixed Points in Tensor Network Renormalization Group]().



## TODO:

- Add links to the LDO paper
- Resolve warning about AbstractAlgebra.mul!
- Resolve warnings in initial_tensor function
- On some computers, sometimes, SVD in TRG step does not converge. That can be fixed by normalizing a tensor before doing SVD. 
