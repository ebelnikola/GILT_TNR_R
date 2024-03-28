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
    help = "If true, the algorithm will perform a rotation by 90 degrees after each GILT TNR step"
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
    "--discrete_gauge_fixing"
    help = "Is true, the algorithm will add discrete gauge fixing to the procedure. This should not change the result."
    arg_type = Bool
    default = false
end

pars = parse_args(settings; as_symbols=true)
for (key, value) in pars
    @eval $key = $value
end


include("../Tools.jl");
include("../GaugeFixing.jl");


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

function gilt_no_dgf(A, pars)
    A, _ = py"gilttnr_step"(A, 1.0, pars)
    A, _ = fix_continuous_gauge(A)
    A = A.to_ndarray()
    return A
end

gilt_tensor = x -> gilt(x, accepted_elements, gilt_pars);

gilt_tensor_no_dgf = x -> gilt_no_dgf(x, gilt_pars);


#############################################################################
# section: FIXING THE NUMBER OF BOND REPETITIONS AND THE RECURSION DEPTH
#############################################################################

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

###############################################
# section: CHOOSING THE DIRECTION FOR DERIVATIVE 
###############################################

function differentiation_direction()
    shape_tmp = A_crit_approximation.shape
    qhape_tmp = A_crit_approximation.qhape
    dirs_tmp = A_crit_approximation.dirs
    array_shape = Tuple(mapslices(sum, shape_tmp; dims=2))
    tmp_ten = zeros(array_shape)
    sector_tmp = (rand([0, 1]), rand([0, 1]), rand([0, 1]), rand([0, 1]))
    while sum(sector_tmp) % 2 != 0
        sector_tmp = (rand([0, 1]), rand([0, 1]), rand([0, 1]), rand([0, 1]))
    end
    ind_tmp = (rand(1:(chi÷2)), rand(1:(chi÷2)), rand(1:(chi÷2)), rand(1:(chi÷2)))
    full_ind_tmp = ((chi ÷ 2) .* sector_tmp) .+ ind_tmp
    tmp_ten[full_ind_tmp...] = 1.0
    return TENSZ2.from_ndarray(tmp_ten, shape=shape_tmp, qhape=qhape_tmp, dirs=dirs_tmp)
end



#############################################
# section: COMPUTE DERIVATIVES
#############################################

samples_of_derivative_computation = Any[]

stp_sizes = 10 .^ (collect(-3:-0.05:-8))

if discrete_gauge_fixing
    fun = gilt_tensor
else
    fun = gilt_tensor_no_dgf
end


function compute_derivatives()
    v = differentiation_direction()

    derivatives = Any[]

    cnt = 0
    for sz in stp_sizes
        try
            push!(derivatives, df(fun, A_crit_approximation, v; order=ord, stp=sz))
        catch e
            push!(derivatives, missing)
        end
        cnt += 1
        println(cnt, " out of ", length(stp_sizes))
    end
    return derivatives
end

for i = 1:N
    push!(samples_of_derivative_computation, compute_derivatives())
end

#############################################
# section: CAUCHY CRITERION
#############################################
fig = Figure(; size=(1000, 1000));

ax = Axis(fig[1, 1],
    yscale=log10,
    xscale=log10,
    xreversed=true,
    xlabel="step size",
    xlabelsize=35,
    xticklabelsize=20,
    ylabel="relative difference between numerical derivatives",
    ylabelsize=35,
    yticklabelsize=20,);

for sample in samples_of_derivative_computation
    lines!(ax, stp_sizes[1:end-1], norm.(variation(sample)) ./ norm.(sample[1:end-1]))
end

fig

name = gilt_pars_identifier(gilt_pars)

save("diff_tests/" * name * "__relT=$(relT)_step=$(number_of_initial_steps).pdf", fig)


