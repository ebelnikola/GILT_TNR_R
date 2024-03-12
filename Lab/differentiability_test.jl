# sections
# - FUNCTIONS FOR TESTING
# - EXPERIMENT INIT
# - CHECKING THE NUMBER OF BOND REPETITIONS AND THE RECURSION DEPTH
# - CHOOSING THE DIRECTION FOR DERIVATIVE
# - TEST FUNCTIONS IN A CONVINIENT FORM
# - COMPUTE DERIVATIVES
# - CAUCHY CRITERION
# - CHECKING THE TRG LOWEST SINGULAR VALUE
# - SPECTRUM PLOTTING SUITE

include("../Tools.jl");
include("../GaugeFixing.jl");


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



################################################
# section: EXPERIMENT INIT
################################################


chi = cla_or_def(1, 10)
gilt_eps = cla_or_def(2, 3e-4)
relT = cla_or_def(3, 0.9975949995592235)
number_of_initial_steps = cla_or_def(4, 20)
order = cla_or_def(5, 2)
verbosity = cla_or_def(6, 3)

cg_eps = cla_or_def(9, 1e-10)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => verbosity,
)

A_crit_approximation = trajectory(relT, number_of_initial_steps, gilt_pars)["A"][end];
A_crit_approximation, Hc, Vc, SHc, SVc = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements = fix_discrete_gauge(A_crit_approximation; tol=1e-7);


#############################################################################
# section: CHECKING THE NUMBER OF BOND REPETITIONS AND THE RECURSION DEPTH
#############################################################################

A1, _ = py"gilttnr_step"(A_crit_approximation, 1.0, gilt_pars);
A1, _ = fix_continuous_gauge(A1);
A1, _ = fix_discrete_gauge(A1, accepted_elements);

(A1 - A_crit_approximation).norm() / A_crit_approximation.norm()

recursion_depth = Dict(
    "S" => 8,
    "N" => 7,
    "E" => 7,
    "W" => 7
)


bond_repetitions = cla_or_def(7, 2)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "bond_repetitions" => bond_repetitions,
    "recursion_depth" => recursion_depth,
)

A2, _ = py"gilttnr_step"(A_crit_approximation, 1.0, gilt_pars);
A2, _ = fix_continuous_gauge(A2);
A2, _ = fix_discrete_gauge(A2, accepted_elements);

(A2 - A_crit_approximation).norm() / A_crit_approximation.norm()



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
    ind_tmp = (rand(1:(chi÷2)), rand(1:(chi÷2)), rand(1:(chi÷2)), rand(1:(chi÷2))) # be careful, the index should be in Z2 conservig sector
    full_ind_tmp = ((chi ÷ 2) .* sector_tmp) .+ ind_tmp
    tmp_ten[full_ind_tmp...] = 1.0
    return TENSZ2.from_ndarray(tmp_ten, shape=shape_tmp, qhape=qhape_tmp, dirs=dirs_tmp)
end


################################################
# section: TEST FUNCTIONS IN A CONVINIENT FORM
################################################

gilt_tensor = x -> gilt(x, accepted_elements, gilt_pars);


#############################################
# section: COMPUTE DERIVATIVES
#############################################

stp_sizes = 10 .^ (collect(-3:-0.05:-8))
fun = gilt_tensor

function compute_derivatives()
    v = differentiation_direction()
    derivatives = Any[]

    cnt = 0
    for sz in stp_sizes
        try
            push!(derivatives, df(fun, A_crit_approximation, v; order=order, stp=sz))
        catch e
            push!(derivatives, missing)
        end
        cnt += 1
        println(cnt, " out of ", length(stp_sizes))
    end
    return derivatives
end

samples_of_derivative_computation = Any[]
for i = 1:1
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



##################################################
# section: CHECKING THE TRG LOWEST SINGULAR VALUE
##################################################

function spectrum(t, v, N)
    pars = deepcopy(gilt_pars)
    pars["cg_chis"] = N
    pars["cg_eps"] = 0
    At = A_crit_approximation + t * v
    A1t, A2t, _ = py"gilt_plaq"(At, At, pars)
    _, _, _, _, SB, SC = py"trg"(A1t, A2t, 1.0, pars)

    SBE = SB.sects[(0,)]
    SBO = SB.sects[(1,)]
    SCE = SC.sects[(0,)]
    SCO = SC.sects[(1,)]

    return SBE, SBO, SCE, SCO
end
v = differentiation_direction()

SBE, SBO, SCE, SCO = spectrum(0, v, chi);



findmin([SBE[end], SBO[end], SCE[end], SCO[end]])



##################################################
# section: SPECTRUM PLOTTING SUITE
##################################################


scale = 1e-4
T = collect(-1:0.01:1) * scale


BE_hist = Vector{Float64}[]
BO_hist = Vector{Float64}[]
CE_hist = Vector{Float64}[]
CO_hist = Vector{Float64}[]

cnt = 0

for t in T
    BE, BO, CE, CO = spectrum(t, v, 24)
    push!(BE_hist, BE[1:11])
    push!(BO_hist, BO[1:11])
    push!(CE_hist, CE[1:11])
    push!(CO_hist, CO[1:11])
    cnt += 1
    println("$cnt out of $(length(T)) are done")
end


BE_hist = vec_of_vec_to_matrix(BE_hist);
BO_hist = vec_of_vec_to_matrix(BO_hist);

CE_hist = vec_of_vec_to_matrix(CE_hist);
CO_hist = vec_of_vec_to_matrix(CO_hist);

fig = Figure(; size=(500, 500));


ax = Axis(fig[1, 1],
    yscale=log10,
)

for i = axes(BE_hist, 1)
    lines!(ax, T, BE_hist[i, :])
end

fig
