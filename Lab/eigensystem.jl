# sections
# - EXPERIMENT INIT
# - RECURSION DEPTH FIXING
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
    default = 10
    "--verbosity"
    help = "Verbosity of the eigensolver"
    arg_type = Int64
    default = 3
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
A_crit_approximation, Hc, Vc, SHc, SVc = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements = fix_discrete_gauge(A_crit_approximation; tol=1e-7);

A_crit_approximation /= A_crit_approximation.norm();
A_crit_approximation_JU = py_to_ju(A_crit_approximation);



################################################
# section: RECURSION DEPTH FIXING AND NORMALISATION
################################################

A1, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars);

tmp = py"depth_dictionary"

recursion_depth = Dict(
    "S" => tmp[(1, "S")],
    "N" => tmp[(1, "N")],
    "E" => tmp[(1, "E")],
    "W" => tmp[(1, "W")]
)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth,
    "rotate" => rotate
)

################################################
# section: FUNCTIONS
################################################

function gilt(A, pars)
    A = ju_to_py(A)
    A, _, _ = py"gilttnr_step"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A, _ = fix_discrete_gauge(A)
    A /= A.norm()
    return py_to_ju(A)
end

function gilt(A, list_of_elements, pars)
    A = ju_to_py(A)
    A, _, _ = py"gilttnr_step"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A, _ = fix_discrete_gauge(A, list_of_elements)
    A /= A.norm()
    return py_to_ju(A)
end


function dgilt(δA)
    return df(x -> gilt(x, accepted_elements, gilt_pars), A_crit_approximation_JU, δA; stp=stp, order=ord)
end


#################################################
# section: EIGENVALUES
#################################################

initial_vector = py_to_ju(random_Z2tens(A_crit_approximation));
res = eigsolve(dgilt, initial_vector, N, :LM; verbosity=verbosity, issymmetric=false, ishermitian=false, krylovdim=30);


result = Dict(
    "A" => A_crit_approximation,
    "eigensystem" => res,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth
)

serialize("eigensystems/" * gilt_pars_identifier(gilt_pars) * "__eigensystem.data", result)

println("EIGENVALUES:")
for val in res[1]
    println(val)
end
