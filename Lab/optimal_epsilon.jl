include("../Tools.jl")


settings = ArgParseSettings()
@add_arg_table! settings begin
    "--chi"
    help = "The bond dimension"
    arg_type = Int64
    default = 10
    "--relT"
    help = "The temperature used in the test"
    arg_type = Float64
    default = 1.0
    "--traj_len"
    help = "The length of trajectory considered in the test"
    arg_type = Int64
    default = 20
    "--cg_eps"
    help = "The threshold used in TRG steps to truncate the bonds"
    arg_type = Float64
    default = 1e-10
    "--eps_low_exponent"
    arg_type = Float64
    default = -8.0
    "--eps_high_exponent"
    arg_type = Float64
    default = -2.0
end

pars = parse_args(settings; as_symbols=true)
for (key, value) in pars
    @eval $key = $value
end


function epsilon_test(gilt_eps, chi=chi, cg_eps=cg_eps, relT=relT)
    gilt_pars = Dict(
        "gilt_eps" => gilt_eps,
        "cg_chis" => collect(1:chi),
        "cg_eps" => cg_eps,
        "verbosity" => 0,
        "rotate" => false
    )
    traj = trajectory_no_saving(relT, traj_len, gilt_pars)
    error_measure = x -> sqrt((x[1])^2 + (x[2] + x[3])^2 / 4 + (x[4] + x[5])^2 / 4)
    findmax(error_measure.(traj["errs"]))[1]
end

@info "Started the test"

epsilons = 10 .^ (collect(eps_high_exponent:-0.1:eps_low_exponent))
results = epsilon_test.(epsilons)

error_measure, index = findmin(results)
optimal_epsilon = epsilons[index]

@info "Found the optimal epsilon = $optimal_epsilon giving the error = $error_measure"

fig = Figure()
ax = Axis(fig[1, 1];
    title="Maximal error along the the trajectory with relT=$relT vs gilt_eps",
    yscale=log10,
    xscale=log10,
    ylabel="maximal error",
    xlabel="gilt_eps"
)

lines!(ax, epsilons, results)

out_path = "out/chi=$chi"
mkpath(out_path)
save(out_path * "/error_vs_gilt_eps.pdf", fig)
serialize(out_path * "/optimal_eps_and_its_error.data", (optimal_epsilon, error_measure))
