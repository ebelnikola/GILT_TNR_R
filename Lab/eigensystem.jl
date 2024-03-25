# sections
# - EXPERIMENT INIT
# - DISCREE GAUGE FIXING MATRICES, RECURSION DEPTH, RMATRICES
# - FUNCTIONS
# - EIGENVALUES


################################################
# section: EXPERIMENT INIT
################################################

using ArgParse

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--chi"
    help = "The bond dimension"
    arg_type = Int64
    default = 10
    "--gilt_eps"
    help = "The threshold used in the GILT algorithm"
    arg_type = Float64
    default = 1e-4
    "--relT"
    help = "The realtive temperature of the initial tensor"
    arg_type = Float64
    default = 1.001277863197029
    "--number_of_initial_steps"
    help = "Number of RG steps made to get an approximation of the critical tensor"
    arg_type = Int64
    default = 15
    "--cg_eps"
    help = "The threshold used in TRG steps to truncate the bonds"
    arg_type = Float64
    default = 1e-10
    "--rotate"
    help = "If true the algorithm will perform a rotation by 90 degrees after each GILT TNR step"
    arg_type = Bool
    default = false
    "--ord"
    help = "order of differentiation algorithm"
    arg_type = Int64
    default = 2
    "--stp"
    help = "step used in differentiation algorithm"
    arg_type = Float64
    default = 5e-6
    "--N"
    help = "Number of eigenvalues to compute"
    arg_type = Int64
    default = 5
    "--verbosity"
    help = "Verbosity of the eigensolver"
    arg_type = Int64
    default = 3
    "--Z2_odd_sector"
    help = "If true the algorithm will find eigenvectors in Z2 breaking sector"
    arg_type = Bool
    default = true
    "--freeze_R"
    help = "If true the algorithm will freeze R matrices."
    arg_type = Bool
    default = false
    "--krylovdim"
    help = "Dimension of the Krylov space used in the eigenvalue computation. This parameter should me larger than N."
    arg_type = Int64
    default = 30
end

pars = parse_args(settings; as_symbols=true)
for (key, value) in pars
    @eval $key = $value
end

include("../Tools.jl");
include("../GaugeFixing.jl");
include("../KrylovTechnical.jl");



gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate
)

A_crit_approximation = trajectory(relT, number_of_initial_steps, gilt_pars)["A"][end];
A_crit_approximation_Z2_broken = A_crit_approximation.to_ndarray();

A_crit_approximation, _ = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements, _ = fix_discrete_gauge(A_crit_approximation; tol=1e-7);
A_crit_approximation /= A_crit_approximation.norm();

A_crit_approximation_JU = py_to_ju(A_crit_approximation);

A_crit_approximation_Z2_broken, _ = fix_continuous_gauge(A_crit_approximation_Z2_broken);
A_crit_approximation_Z2_broken, accepted_elements_Z2_broken, _, _ = fix_discrete_gauge(A_crit_approximation_Z2_broken);
normalize!(A_crit_approximation_Z2_broken);

#####################################################################
# section: DISCREE GAUGE FIXING MATRICES, RECURSION DEPTH, RMATRICES
#####################################################################


py"""
def gilttnr_step_broken(A,log_fact,pars):
    A=Tensor.from_ndarray(A)
    if "Rmatrices" in pars:
        for key,value in  pars["Rmatrices"].items():
            pars["Rmatrices"][key]=Tensor.from_ndarray(value)
    return gilttnr_step(A,log_fact,pars)
"""

if Z2_odd_sector
    Atmp, _ = py"gilttnr_step_broken"(A_crit_approximation_Z2_broken, 0.0, gilt_pars)
    Atmp, _ = fix_continuous_gauge(Atmp)
    Atmp, _, H, V = fix_discrete_gauge(Atmp, accepted_elements_Z2_broken)
else
    Atmp, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars)
    Atmp, _ = fix_continuous_gauge(Atmp)
    Atmp, _, H, V = fix_discrete_gauge(Atmp, accepted_elements)
end

Rmatrices = py"Rmatrices";
tmp = py"depth_dictionary"
recursion_depth = Dict(
    "S" => tmp[(1, "S")],
    "N" => tmp[(1, "N")],
    "E" => tmp[(1, "E")],
    "W" => tmp[(1, "W")]
)

if freeze_R
    gilt_pars = Dict(
        "gilt_eps" => gilt_eps,
        "cg_chis" => collect(1:chi),
        "cg_eps" => cg_eps,
        "verbosity" => 0,
        "bond_repetitions" => 2,
        "Rmatrices" => Rmatrices,
        "rotate" => rotate
    )
else
    gilt_pars = Dict(
        "gilt_eps" => gilt_eps,
        "cg_chis" => collect(1:chi),
        "cg_eps" => cg_eps,
        "verbosity" => 0,
        "bond_repetitions" => 2,
        "recursion_depth" => recursion_depth,
        "rotate" => rotate
    )
end

################################################
# section: FUNCTIONS
################################################

function gilt(A, pars)
    A = ju_to_py(A)
    A, _, _ = py"gilttnr_step"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A = ncon([A, H.conj(), V.conj(), H, V], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    A /= A.norm()
    return py_to_ju(A)
end

function gilt(A::Array, pars)
    A, _, _ = py"gilttnr_step_broken"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A = ncon([A, H, V, H, V], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    normalize!(A)
    return A
end

function dgilt(δA)
    return df(x -> gilt(x, gilt_pars), A_crit_approximation_JU, δA; stp=stp, order=ord)
end

function dgilt(δA::Array)
    return df(x -> gilt(x, gilt_pars), A_crit_approximation_Z2_broken, δA; stp=stp, order=ord)
end


#################################################
# section: EIGENVALUES
#################################################


if Z2_odd_sector
    initial_vector = normalize!(2 .* rand(chi, chi, chi, chi) .- 1)
else
    initial_vector = py_to_ju(random_Z2tens(A_crit_approximation))
end;

res = eigsolve(dgilt, initial_vector, N, :LM; verbosity=verbosity, issymmetric=false, ishermitian=false, krylovdim=krylovdim);

if freeze_R
    result = Dict(
        "A" => A_crit_approximation,
        "eigensystem" => res,
        "bond_repetitions" => 2,
        "Rmatrices" => Rmatrices
    )
else
    result = Dict(
        "A" => A_crit_approximation,
        "eigensystem" => res,
        "bond_repetitions" => 2,
        "recursion_depth" => recursion_depth
    )
end

serialize("eigensystems/" * gilt_pars_identifier(gilt_pars) * "__Z2_odd_sector=$(Z2_odd_sector)_freeze_R=$(freeze_R)_eigensystem.data", result)

println("EIGENVALUES:")
for val in res[1]
    println(val)
end



