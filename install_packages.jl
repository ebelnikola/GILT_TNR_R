# The process may take some time

using Pkg
Pkg.add("PyCall")
Pkg.add("LinearAlgebra")
Pkg.add("Serialization")
Pkg.add("CairoMakie")
Pkg.add("ArgParse")
Pkg.add("AbstractAlgebra")
Pkg.add("KrylovKit")

mkpath("critical_temperatures")
mkpath("diff_tests")
mkpath("eigensystems")
mkpath("newton")
mkpath("trajectories")
mkpath("trajectory_plots")
