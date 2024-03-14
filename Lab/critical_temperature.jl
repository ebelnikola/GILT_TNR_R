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
    "--relT_low"
    help = "The lower bound for the critical temperature search"
    arg_type = Float64
    default = 0.99
    "--relT_high"
    help = "The higher bound for the critical temperature search"
    arg_type = Float64
    default = 1.01
    "--search_tol"
    help = "Precision with which the critical temperature should be found"
    arg_type = Float64
    default = 1e-5
    "--max_number_of_steps"
    help = "The maximal number of steps of GILT TNR that will be done in order to differntiate between phases"
    arg_type = Int64
    default = 100
    "--cg_eps"
    help = "The threshold used in TRG steps to truncate the bonds"
    arg_type = Float64
    default = 1e-10
    "--rotate"
    help = "If true the algorithm will perform a rotation by 90 degrees after each GILT TNR step"
    arg_type = Bool
    default = false
end

pars = parse_args(settings; as_symbols=true)
for (key, value) in pars
    @eval $key = $value
end

include("../Tools.jl")


const global gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 3,
    "rotate" => rotate
)

gilt_pars["verbosity"] = 0

search_result = perform_search(relT_low, relT_high, gilt_pars; search_tol=search_tol, max_number_of_steps=max_number_of_steps, verbose=false)

serialize("critical_temperatures/" * gilt_pars_identifier(gilt_pars) * "__tol=$(search_tol).data", search_result)

gilt_pars["verbosity"] = 3

plot_the_trajectory((search_result[1] + search_result[2]) / 2, gilt_pars; traj_len=search_result[3])
