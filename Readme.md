# Naming convention

The method of work with the database in this repository is not ideal and, perhaps, will be revisited in the future to eliminate the possibility of introducing ambiguities in the database.   

Most of the data generated by the codes provided in this repository will be saved using the following naming convention.
```
rotate=[rotate]_[chi]_[gilt_eps]_[cg_eps]__[info about particular experiment].[extension]
```
Here, the first part of the name ```rotate=[rotate]_[chi]_[gilt_eps]_[cg_eps]``` characterizes the GiltTNR algorithm uniquely up to the following ambiguity. This convention assumes that no ```bond_repetitions```, ```recursion_depth```, or ```Rmatrices``` were provided (see the list of adjustments in the GiltTNR directory section). There are two functions generating entries in the database using this convention: ```trajectory``` and ```plot_the_trajectory``` (in particular their methods with ```relT``` argument). Thus, the user should avoid passing ```bond_repetitions```, ```recursion_depth```, or ```Rmatrices``` into these functions as this will cause ambiguities in the database.    

# Files and directories descriptions

## GiltTNR directory

This directory contains Python codes performing the GiltTNR algorithm (see [this paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111)). It is an assembly of codes from [GiltTNR](https://github.com/GiltTNR/GiltTNR), [tensorRGflow](https://github.com/brucelyu/tensorRGflow), [ncon](https://github.com/mhauru/ncon), [abeliantensors](https://github.com/mhauru/abeliantensors). We applied minor adjustments:

1. We added an option to fix the number of iterations of the Gilt algorithm applied to each bond around a plaquette. It can be fixed by passing a keyword argument ```bond_repetitions```. 
2. We added an option to fix the recursion depth in the optimization procedure that searches for the ```R``` matrix (for the ```R``` matrix definition see [this paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111)). It can be fixed by passing a keyword argument ```recursion_depth``` (together with ```bond_repetitions```) that is a dictionary whose keys belong to ```{"S", "N", "E", "W"}```, and whose values are the recursion depths that should be used at the corresponding leg. Note that at the last Gilt iteration, recursion depth will be automatically set to 1 for all legs as this is what typically happens when the standard convergence stopping criteria is used.    
3. We added an option to use precomputed ```R``` matrices instead of running the optimization procedure. For this one should pass ```bond_repetitions``` and the keyword argument ```Rmatrices``` which is a dictionary whose keys are the pares ```(bond_key, lap)``` (```lap``` is an integer,```bond_key in {"S", "N", "E", "W"}```) and whose values are the ```R``` matrices which should be applied at the bond ```bond_key``` at ```lap``` iteration of the Gilt.
4. We added an option to choose whether to rotate the tensor after the GiltTNR step or not. The keyword ```rotate``` controls this.    
5. We modified ```matrix_eig``` and ```matrix_svd``` functions in ```abeliantensor.py``` so that the sign ambiguities in the decomposition are removed.  

## Lab directory

Contains scripts for computations:

- ```plot_trajectory.jl``` applies the ```GiltTNR``` algorithm ```traj_len``` times to the initial tensor corresponding to 2d nearest neighbors Ising model at the relative temperature ```relT```. It saves the resulting trajectories to the "trajectories" folder in two files: ```*.data``` with tensors, log factors, and errors along the trajecotry; ```*.log``` with all the text output of the algorithm. It also saves the plots of the trajectories of the tensor's singular values (obtained by "diagonal" SVD: $A_{ijkl}=U_{ij l} S_l V_{l kl}$).

- ```critical_temperature.jl``` finds the critical temperature using bisection search. Saves the result into the "critical_temperatures" directory. Also plots the corresponding trajectories of the singular values. Saves the plot into the "trajectory_plots" directory. 
 
- ```differentiability_test.jl``` performs differentiability tests of the GiltTNR algorithm. Chooses a random direction $v$ and computes the numerical derivative $\Delta_n=\frac{GiltTNR(A+s_n*v/2)-GiltTNR(A-s_n*v/2)}{s_n}$, where $s_n=10^{-3-0.05n}$. Plots $|\Delta_{n+1}-\Delta_n|/|\Delta_n|$ vs $n$ and saves this plot into the "results" directory. The approximate minimum of this plot should be chosen as the step size in ```newton.jl```. If parameter ```N``` is passed, will perform test for ```N``` random directions. 

- ```eigensystem.jl``` gets the largest eigenvalues and the corresponding eigenvectors of GiltTNR linearised around some initial approximation of the critical tensor (given by ```relT``` and ```number_of_initial_steps```). Saves the resulting tensor and the eigensystem to ```eigensystems```. 

- ```newton.jl``` assumes that ```rotation=true```. Repeats the computation from ```eigensystem.jl```. Then, finds the critical tensor using Newton's method and computes the eigensystem for the linearisation of GiltTNR around this tensor. Saves the resulting tensor together with the eigensystem to the "newton" directory. Also saves Newton's method convergence plot to the same directory.

Note that each of these scripts has the corresponding help describing all the command line arguments. To see this help run 
```
julia Lab/script_name.jl --help
```

## Files in the root directory

- ```EchelonForm.jl``` - technical code used in ```GaugeFixing.jl``` for checking ranks of boolean matrices. 
- ```GaugeFixing.jl``` - continuous and discrete gauge fixing routines.
- ```install_packages.jl``` - script that installs all the necessary dependencies and creates necessary directories.
- ```KrylovTechnical.jl``` - technical code that provides a minimal implementation of Z2 invariant tensors in Julia. This solves the problem of ```KrylovKit``` throwing a segmentation fault while working with PyObjects.
- ```NumDifferentiation.jl``` - provides the function that performs numerical differentiation.
- ```Tools.jl``` - Contains user-dedicated functions. See the top of the file for the list of functions.    
- ```IsingExactLevels``` and ```IsingEvenExactLevels``` - data about exact 2d Ising spectrum.
- ```Lab.ipynb``` - the notebook that performs all the same computations as the scripts in the "Lab" directory.  

# Performing the computations
 
Ensure that your working directory is the root directory of this project. Then, run the ```install_packages.jl``` script to install all the necessary dependencies: 
```
   julia install_packages.jl
```

## Eigenvalues of GiltTNR without rotation at an approximate fixed point

Let us do the computation for $\chi=10$, $gilt\_eps=1e-4$. To get an approximate fixed point we run the bisection search:
```
nohup julia Lab/critical_temperature.jl --chi 10 --gilt_eps 1e-4 --search_tol 1e-10 &
```

The script will produce the singular values trajectory plot corresponding to the found approximation of the critical temperature. The plot exhibits a plateau corresponding to the part of the trajectory close to the critical tensor. We will choose one tensor at this plateau as an approximation of the critical tensor and will compute eigenvalues of GiltTNR by linearisation around this point. Let us take the tensor obtained at step 15. Before computing the eigensystem it is useful to run the differentiability test and check what should be the optimal differentiation step. Here is the command for this:  

```
nohup julia Lab/differentiability_test.jl --chi 10 --gilt_eps 1e-4 --relT 1.001277863197029 --number_of_initial_steps 15 --N 5 &
```

Note that ```differentiability_test``` will fix the ```bond_repetitions``` and ```recursion_depth```. This should ensure that the resulting RG map is differentiable. As it is written now, the script will fix ```bond_repetitions=2```, as for the ```recursion_depth``` it will read off the recursion depths that are used in GiltTNR before these manipulations and use these values.

Checking the Cauchy test plots in the "diff_tests" folder we see that derivatives do converge. The slope of the lines is around $-2$ which is consistent with the numerical differentiation algorithm of order $2$ that is used in the script by default. We also see that stepsize $5e-6$ should be an appropriate choice.  

Finally, we may run ```eigensystem.jl``` (no "nohup" this time as it is nice to see the correct eigenvalues printed out in the terminal):
```
julia Lab/eigensystem.jl --chi 10 --gilt_eps 1e-4 --relT 1.001277863197029 --number_of_initial_steps 15 --stp 5e-6 --N 20
```
## Critical fixed point of GiltTNR with rotation

To be written 



