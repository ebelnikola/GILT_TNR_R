include("../Tools.jl")


const global chi = cla_or_def(1, 10)
const global gilt_eps = cla_or_def(2, 3e-5)

const global relT_low = cla_or_def(3, 0.99)
const global relT_high = cla_or_def(4, 1.01)
const global search_tol = cla_or_def(5, 1e-10)
const global max_number_of_steps = cla_or_def(6, 50)


const global verbosity = cla_or_def(7, 2)
const global cg_eps = cla_or_def(8, 1e-10)


const global gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => verbosity,
)


search_result = perform_search(relT_low, relT_high, gilt_pars; search_tol=search_tol, max_number_of_steps=max_number_of_steps, verbose=false)

serialize("critical_temperatures/" * gilt_pars_identifier(gilt_pars) * "__tol=$(search_tol).data", search_result)

plot_the_trajectory(;
    chi=chi,
    relT=(search_result[1] + search_result[2]) / 2,
    traj_len=search_result[3],
    gilt_eps=gilt_eps,
    verbosity=2
)

