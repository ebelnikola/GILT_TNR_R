# sections
# - FUNCTIONS FOR TESTING
# - EXPERIMENT INIT
# - CHECKING THE NUMBER OF BOND REPETITIONS AND THE RECURSION DEPTH
# - CHOOSING THE DIRECTION FOR DERIVATIVE
# - TEST FUNCTIONS IN A CONVINIENT FORM
# - COMPUTE DERIVATIVES
# - CAUCHY CRITERION


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
	help = "The realtive temperature of the initial tensor"
	arg_type = Float64
	default = 1.0000110043212773
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
	help = "Order of differentiation algorithm"
	arg_type = Int64
	default = 2
	"--N"
	help = "Number of randomised tests"
	arg_type = Int64
	default = 2
	"--largest_step_exponent"
	help = "The code will take derivatives of the Gilt procedure using step sizes below 10^(largest_step_exponent)"
	arg_type = Float64
	default = -1.5
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

initialA_pars = Dict("relT" => relT, "Jratio" => Jratio)
A_crit_approximation = trajectory(initialA_pars, number_of_initial_steps, gilt_pars)["A"][end];
A_crit_approximation, Hc, Vc, SHc, SVc = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements = fix_discrete_gauge(A_crit_approximation; tol = 1e-7);

A_crit_approximation = A_crit_approximation / A_crit_approximation.norm();



#############################################################################
# section: FIXING THE NUMBER OF BOND REPETITIONS AND THE RECURSION DEPTH
#############################################################################

A1, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars);

tmp = py"depth_dictionary";

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
# section: FUNCTIONS FOR TESTING
################################################

function gilt(A, pars)
	A, _ = py"gilttnr_step"(A, 1.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A)
	A = A.to_ndarray()
	return A
end

function gilt(A, list_of_elements, pars)
	A, _ = py"gilttnr_step"(A, 1.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A, list_of_elements)
	A = A.to_ndarray()
	return A
end

gilt_tensor = x -> gilt(x, accepted_elements, gilt_pars);



#############################################
# section: COMPUTE DERIVATIVES
#############################################

stp_sizes = 10 .^ (collect(largest_step_exponent:-0.05:-8))
fun = gilt_tensor

function compute_derivatives()
	v = random_Z2tens(A_crit_approximation)
	derivatives = Any[]

	cnt = 0
	for sz in stp_sizes
		try
			push!(derivatives, df(fun, A_crit_approximation, v; stp = sz, order = ord))
		catch e
			push!(derivatives, missing)
		end
		cnt += 1
		println(cnt, " out of ", length(stp_sizes))
	end
	return derivatives
end

function compute_derivatives(v)
	derivatives = Any[]

	cnt = 0
	for sz in stp_sizes
		try
			push!(derivatives, df(fun, A_crit_approximation, v; stp = sz, order = ord))
		catch e
			push!(derivatives, missing)
		end
		cnt += 1
		println(cnt, " out of ", length(stp_sizes))
	end
	return derivatives
end


samples_of_derivative_computation = Any[]

for i ∈ 1:N
	push!(samples_of_derivative_computation, compute_derivatives())
end


#=
v = real(deserialize("5e-6_step_eigenvector.data")[2][1] |> py_to_ju);
v = ju_to_py(v);


eigenvalue2derivatives = compute_derivatives(v)
random_dir = compute_derivatives()

output = Dict(
	"steps" => stp_sizes,
	"eigenvalue_2_direction" => eigenvalue2derivatives,
	"random_direction" => random_dir,
	"rec_depth" => recursion_depth,
);

serialize("chi=30_computed_derivatives_stp=5e-6.data", eigenvalue2derivatives)
=#
#############################################
# section: CAUCHY CRITERION
#############################################
fig = Figure(; size = ((1 + sqrt(5)) * 200, 400));

ax = Axis(fig[1, 1],
	yscale = log10,
	xscale = log10,
	xreversed = true,
	xlabel = L"\text{step size}",
	ylabel = L"\text{relative difference between numerical derivatives}",
);


for sample in samples_of_derivative_computation
	lines!(ax, stp_sizes[6:end-1], (norm.(variation(sample))./norm.(sample[1:end-1]))[6:end])
end

fig


name = gilt_pars_identifier(gilt_pars)

save("diff_tests/" * name * "__relT=$(relT)_step=$(number_of_initial_steps).pdf", fig)

