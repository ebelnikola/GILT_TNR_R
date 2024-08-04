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
	default = 30
	"--gilt_eps"
	help = "The threshold used in the GILT algorithm"
	arg_type = Float64
	default = 6e-6
	"--relT"
	help = "The relative temperature of the initial tensor"
	arg_type = Float64
	default = 0.0
	"--Jratio"
	help = "The anisotropy parameter of the initial tensor"
	arg_type = Float64
	default = 1.0
	"--number_of_initial_steps"
	help = "Number of RG steps made to get an approximation of the critical tensor"
	arg_type = Int64
	default = 23
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
	default = 1e-4
	"--N"
	help = "Number of eigenvalues to compute"
	arg_type = Int64
	default = 10
	"--verbosity"
	help = "Verbosity of the eigensolver"
	arg_type = Int64
	default = 0
	"--Z2_odd_sector"
	help = "If true the algorithm will find eigenvectors in Z2 breaking sector"
	arg_type = Bool
	default = true
	"--freeze_R"
	help = "If true the algorithm will freeze R matrices."
	arg_type = Bool
	default = false
	"--krylovdim"
	help = "Dimension of the Krylov space used in the eigenvalue computation. This parameter should be larger than N."
	arg_type = Int64
	default = 30
	"--path_to_tensor"
	help = "If provided, the code will ignore relT, Jratio, and N and will load the initial tensor from the given path. The file with the tensor should be created using Julia's serialize function. It should be a dictionary that contains the tensor A under key \"A\" and the recursion depths array under key \"recursion_depth\"."
	arg_type = String
	default = "none"
end

pars = parse_args(settings; as_symbols = true)
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
	"rotate" => rotate,
)

if path_to_tensor == "none"
	if relT == 0 # then we search in critical_temperatures/ to find critical relT
		# if case is not found in critical_temperatures we throw an exception
		search_tol = 1.0e-10
		relT_low, relT_high, relT = find_critical_temperature(chi, gilt_pars, search_tol, Jratio)
		if abs(relT) < 1.e-10
			throw("Critical relT was not found in critical_temperatures/")
		end
	end

	initialA_pars = Dict("relT" => relT, "Jratio" => Jratio)

	A_crit_approximation = trajectory(initialA_pars, number_of_initial_steps, gilt_pars)["A"][end]
else
	tensor_data = deserialize(path_to_tensor)
	A_crit_approximation = tensor_data["A"]
end

A_crit_approximation_Z2_broken = A_crit_approximation.to_ndarray();

A_crit_approximation, _ = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements, _ = fix_discrete_gauge(A_crit_approximation; tol = 1e-7);
A_crit_approximation /= A_crit_approximation.norm();


A_crit_approximation_JU = py_to_ju(A_crit_approximation);

ZH = diagm(vcat(ones(chi ÷ 2), -ones(chi ÷ 2)))
ZV = diagm(vcat(ones(chi ÷ 2), -ones(chi ÷ 2)))

A_crit_approximation_Z2_broken, GH, GV, SH, SV = fix_continuous_gauge(A_crit_approximation_Z2_broken);
A_crit_approximation_Z2_broken, accepted_elements_Z2_broken, _, _ = fix_discrete_gauge(A_crit_approximation_Z2_broken);
normalize!(A_crit_approximation_Z2_broken);

ZH = GH' * ZH * GH
ZV = GV' * ZV * GV

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
end;


Rmatrices = py"Rmatrices"
tmp = py"depth_dictionary"


if path_to_tensor == "none"
	recursion_depth = Dict(
		"S" => tmp[(1, "S")],
		"N" => tmp[(1, "N")],
		"E" => tmp[(1, "E")],
		"W" => tmp[(1, "W")],
	)
else
	recursion_depth = tensor_data["recursion_depth"]
end

if freeze_R
	gilt_pars = Dict(
		"gilt_eps" => gilt_eps,
		"cg_chis" => collect(1:chi),
		"cg_eps" => cg_eps,
		"verbosity" => 0,
		"bond_repetitions" => 2,
		"Rmatrices" => Rmatrices,
		"rotate" => rotate,
	)
else
	gilt_pars = Dict(
		"gilt_eps" => gilt_eps,
		"cg_chis" => collect(1:chi),
		"cg_eps" => cg_eps,
		"verbosity" => 0,
		"bond_repetitions" => 2,
		"recursion_depth" => recursion_depth,
		"rotate" => rotate,
	)
end


################################################
# section: FUNCTIONS
################################################

function gilt(A, pars)
	A = ju_to_py(A)
	A, _, _ = py"gilttnr_step"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A, accepted_elements)
	#A = ncon([A, H.conj(), V.conj(), H, V], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
	A /= A.norm()
	return py_to_ju(A)
end

function gilt(A::Array, pars)
	A, _, _ = py"gilttnr_step_broken"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A, accepted_elements_Z2_broken)
	#A = ncon([A, H, V, H, V], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
	normalize!(A)
	return A
end

function dgilt(δA)
	return df(x -> gilt(x, gilt_pars), A_crit_approximation_JU, δA; stp = stp, order = ord)
end

function dgilt(δA::Array)
	return df(x -> gilt(x, gilt_pars), A_crit_approximation_Z2_broken, δA; stp = stp, order = ord)
end

#################################################
# section: EIGENVALUES
#################################################

if Z2_odd_sector
	initial_vector = normalize!(2 .* rand(chi, chi, chi, chi) .- 1)
else
	initial_vector = py_to_ju(random_Z2tens(A_crit_approximation))
end;

res = eigsolve(dgilt, initial_vector, N, :LM; verbosity = verbosity, issymmetric = false, ishermitian = false, krylovdim = krylovdim, maxiter = 200);

if freeze_R
	result = Dict(
		"A" => A_crit_approximation,
		"eigensystem" => res,
		"bond_repetitions" => 2,
		"Rmatrices" => Rmatrices,
	)
else
	result = Dict(
		"A" => A_crit_approximation,
		"eigensystem" => res,
		"bond_repetitions" => 2,
		"recursion_depth" => recursion_depth,
	)
end


filename = gilt_pars_identifier(gilt_pars) * "__Z2_odd_sector=$(Z2_odd_sector)_freeze_R=$(freeze_R)"

# We only include Jratio in the file name if Jratio is not 1
# This insures backwards compatibility with earlier file names
if abs(Jratio - 1) > 1.e-10
	filename = filename * "_Jratio=$(Jratio)"
end

if path_to_tensor != "none"
	filename *= ("_path=" * path_to_tensor)
end

filename = filename * ".data"

serialize("eigensystems/" * filename, result)

print("EIGENVALUES for : Jratio=", Jratio, " chi=", chi, " freeze=", freeze_R)
println("  gilt_eps=", gilt_eps, "  cg_eps=", cg_eps)

gilt_eps = gilt_pars["gilt_eps"]
cg_eps = gilt_pars["cg_eps"]

for i in range(1, length(res[1]))
	val = res[1][i]
	println(val, " |  Z2 q.n.=", parity(res[2][i], ZH, ZV))
end

#=
# print complex eigenvalue
for i in range(1, length(res[1]))
	val = res[1][i]
	println(Jratio, " : ", val, " | evalonly evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end


# print real part of eigenvalue
for i in range(1, length(res[1]))
	val = real(res[1][i])
	println(Jratio, "  ", val, " | evalreal evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end

# print abs of eigenvalue
for i in range(1, length(res[1]))
	val = abs(res[1][i])
	println(Jratio, "  ", val, " | evalabs evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end

# print log of abs of eigenvalue
for i in range(1, length(res[1]))
	val = log(abs(res[1][i]))
	println(Jratio, "  ", val, " | evallog evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end
=#