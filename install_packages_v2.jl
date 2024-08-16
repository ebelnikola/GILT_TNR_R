# The process may take some time

using Pkg
Pkg.add("IJulia") # adds IJulia globally
Pkg.activate(".")
Pkg.add("IJulia") # adds IJulia to this project 
Pkg.add("LinearAlgebra")
Pkg.add("Serialization")
Pkg.add(name = "PyCall", version = "1.96.4")
Pkg.add(name = "CairoMakie", version = "0.12.2")
Pkg.add(name = "ArgParse", version = "1.2.0")
Pkg.add(name = "AbstractAlgebra", version = "0.41.8")
Pkg.add(name = "KrylovKit", version = "0.6.1")
Pkg.add(name = "TensorOperations", version = "4.1.1")
Pkg.add(name = "DataFrames", version = "1.6.1")
Pkg.add(name = "Colors", version = "0.12.11")
Pkg.add(name = "LaTeXStrings", version = "1.3.1")

