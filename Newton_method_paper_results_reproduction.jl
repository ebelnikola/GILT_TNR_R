
println("This script will reproduce some of the results presented in the paper: \"Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group\"")

println("It will take some time. Depending on a machine some cripts may run from 20 minutes to 1.5 hour.")

println("Let us start with adding all the necessary dependences.")


include("install_packages.jl")
include("Tools.jl")
include("KrylovTechnical.jl")
include("GaugeFixing.jl")


println("Dependences are added. Now, let us reproduce Figure 4 from the paper.")
println("For this we run the critical_temperature.jl script with default values by calling \"julia --project Lab/critical_temperature.jl\"")
println("The script will save Figure 4 in trajectory_plots folder")
println("\"--project\" flag here tells Julia to use the Project files specified in the root of this repository.")

run(`julia --project Lab/plot_trajectory.jl`)