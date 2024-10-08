# sections
# - EXPERIMENT INIT
# - RECURSION DEPTH FIXING
# - FUNCTIONS
# - NEWTON
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
	help = "The threshold used in the Gilt algorithm"
	arg_type = Float64
	default = 6e-6
	"--relT"
	help = "The realtive temperature of the initial tensor"
	arg_type = Float64
	default = 0.0
	"--Jratio"
	help = "The anisotropy parameter"
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
	"--ord"
	help = "order of differentiation algorithm"
	arg_type = Int64
	default = 2
	"--stp"
	help = "step used in differentiation algorithm"
	arg_type = Float64
	default = 1e-4
	"--eigensystem_size_for_jacobian"
	help = "Number of eigenvectors and eigenvalues to be used to approximate the jacobian"
	arg_type = Int64
	default = 9
	"--N"
	help = "Number of steps performed by the Newton algorithm"
	arg_type = Int64
	default = 32
	"--verbosity"
	help = "Verbosity of the eigensolver"
	arg_type = Int64
	default = 0
end

const global rotate = true

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

# if relT==0 then we search in critical_temperatures/ to find critical relT
# if case is not found in critical_temperatures we throw an exception
if relT == 0
	search_tol = 1.0e-10
	relT_low, relT_high, relT = find_critical_temperature(chi, gilt_pars, search_tol, Jratio)
	if abs(relT) < 1.e-10
		throw("Critical relT was not found in critical_temperatures/")
	end
end

initialA_pars = Dict("relT" => relT, "Jratio" => Jratio)
A_crit_approximation = trajectory(initialA_pars, number_of_initial_steps, gilt_pars)["A"][end];
A_crit_approximation, Hc, Vc, SHc, SVc = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements = fix_discrete_gauge(A_crit_approximation; tol = 1e-7);

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
	"W" => tmp[(1, "W")],
)

gilt_pars = Dict(
	"gilt_eps" => gilt_eps,
	"cg_chis" => collect(1:chi),
	"cg_eps" => cg_eps,
	"verbosity" => 0,
	"bond_repetitions" => 2,
	"recursion_depth" => recursion_depth,
	"rotate" => rotate,
)

################################################
# section: FUNCTIONS
################################################
function gilt(A, pars)
	A = ju_to_py(A)
	A, _ = py"gilttnr_step"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A)
	A /= A.norm()
	return py_to_ju(A)
end

function gilt(A, list_of_elements, pars)
	A = ju_to_py(A)
	A, _ = py"gilttnr_step"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A, list_of_elements)
	A /= A.norm()
	return py_to_ju(A)
end


function dgilt(δA)
	return df(x -> gilt(x, accepted_elements, gilt_pars), A_crit_approximation_JU, δA; stp = stp, order = ord)
end


function build_jacobian_approximation(vectors::Vector{t}, values::Vector) where {t}
	rank = length(values)
	jacobian_approximation = zeros(rank, rank)
	basis = Vector{t}(undef, rank)
	i = 1
	while i <= rank
		if imag(values[i]) == 0
			jacobian_approximation[i, i] = real(values[i])
			basis[i] = real(vectors[i])
			i += 1
		else
			if conj(values[i+1]) != values[i]
				throw("Unmatched complex eigenvlaue is detected")
			end
			λ₁ = real(values[i])
			λ₂ = imag(values[i])
			v1 = real(vectors[i])
			v2 = imag(vectors[i])
			v1norm = v1 |> norm
			v2norm = v2 |> norm
			v1 /= v1norm
			v2 /= v2norm
			jacobian_approximation[i, i] = λ₁
			jacobian_approximation[i+1, i+1] = λ₁
			jacobian_approximation[i+1, i] = -λ₂ * v2norm / v1norm
			jacobian_approximation[i, i+1] = λ₂ * v1norm / v2norm
			basis[i] = v1
			basis[i+1] = v2
			i += 2
		end
	end
	jacobian_approximation, basis
end

function build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis::Vector{t}) where {t}
	dim = length(non_orthogonal_normalised_basis)
	Graham_Schmidt_Matrix = zeros(dim, dim)
	Graham_Schmidt_Matrix[1, 1] = 1
	orthonormal_basis = t[non_orthogonal_normalised_basis[1]]
	for n ∈ 1:(dim-1)
		new_vector = non_orthogonal_normalised_basis[n+1]
		for i ∈ 1:n
			new_vector -= dot(orthonormal_basis[i], new_vector) * orthonormal_basis[i]
		end
		Nnp1 = norm(new_vector)
		new_vector /= Nnp1
		push!(orthonormal_basis, new_vector)

		gnn = Graham_Schmidt_Matrix[1:n, 1:n]
		v_old_dot_v_new_vector = zeros(n)
		for i ∈ 1:n
			v_old_dot_v_new_vector[i] = dot(non_orthogonal_normalised_basis[i], non_orthogonal_normalised_basis[n+1])
		end
		Graham_Schmidt_Matrix[1:n, n+1] .= -1 / Nnp1 .* gnn * transpose(gnn) * v_old_dot_v_new_vector
		Graham_Schmidt_Matrix[n+1, n+1] = 1 / Nnp1
	end
	return Graham_Schmidt_Matrix, orthonormal_basis
end


#################################################
# section: NEWTON
#################################################
@time begin
	initial_vector = py_to_ju(random_Z2tens(A_crit_approximation))
	eigensystem_init = eigsolve(dgilt, initial_vector, eigensystem_size_for_jacobian, :LM; verbosity = verbosity, issymmetric = false, ishermitian = false, krylovdim = eigensystem_size_for_jacobian + 20)
end
println("EIGENVALUES (INITIAL):")
for val in eigensystem_init[1]
	println(val)
end

if length(eigensystem_init[1]) > eigensystem_size_for_jacobian
	if conj(eigensystem_init[1][eigensystem_size_for_jacobian]) ≈ eigensystem_init[1][eigensystem_size_for_jacobian+1]
		approximation_rank = eigensystem_size_for_jacobian + 1
	else
		approximation_rank = eigensystem_size_for_jacobian
	end
else
	approximation_rank = eigensystem_size_for_jacobian
end

eigensystem_init = [eigensystem_init[1][1:approximation_rank], eigensystem_init[2][1:approximation_rank]];


jac_approximation_non_orthogonal_basis, non_orthogonal_normalised_basis = build_jacobian_approximation(eigensystem_init[2], eigensystem_init[1]);
Graham_Schmidt_matrix, orthonormal_basis = build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis);
jac_approximation = Graham_Schmidt_matrix^(-1) * jac_approximation_non_orthogonal_basis * Graham_Schmidt_matrix;


ImJ_inv_matrix = (I - jac_approximation)^(-1);


function project_to_Vs(δA)
	res = zero(δA)
	for i in 1:approximation_rank
		res += orthonormal_basis[i] * dot(orthonormal_basis[i], δA)
	end
	return res
end

function ImJ_inv(δA)
	δA_in_Vs = project_to_Vs(δA)
	ImJ_inv_δA_in_Vs = zero(δA)
	δA_out_of_Vs = δA - δA_in_Vs
	for i ∈ 1:approximation_rank
		for j ∈ 1:approximation_rank
			ImJ_inv_δA_in_Vs += ImJ_inv_matrix[i, j] * orthonormal_basis[i] * dot(orthonormal_basis[j], δA)
		end
	end
	return ImJ_inv_δA_in_Vs + δA_out_of_Vs
end


function newton_function(A)
	x_minus_f = A - gilt(A, accepted_elements, gilt_pars)
	correction = ImJ_inv(x_minus_f)
	return A - correction
end

A_hist = [A_crit_approximation_JU];

@time begin
	for i ∈ 2:N
		push!(A_hist, newton_function(A_hist[i-1]))
		println("Newton step size: $(A_hist[i] - A_hist[i-1] |> norm)")
	end
end

result = Dict(
	"A_newton" => A_hist,
	"eigensystem_init" => eigensystem_init,
	"bond_repetitions" => 2,
	"recursion_depth" => recursion_depth,
)

serialize("newton/" * gilt_pars_identifier(gilt_pars) * "_jac_approximation_rank=$(approximation_rank).data", result)




