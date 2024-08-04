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
	"--relT_low"
	help = "The lower bound for the critical temperature search"
	arg_type = Float64
	default = 0.99
	"--relT_high"
	help = "The higher bound for the critical temperature search"
	arg_type = Float64
	default = 1.01
	"--Jratio"
	help = "The anisotropy parameter"
	arg_type = Float64
	default = 1.0
	"--search_tol"
	help = "Precision with which the critical temperature should be found"
	arg_type = Float64
	default = 1e-10
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
	"--plot_trajectory"
	help = "If true, the trajectory corresponding to the found critical temperature will be plotted"
	arg_type = Bool
	default = true
end

pars = parse_args(settings; as_symbols = true)
for (key, value) in pars
	@eval $key = $value
end

include("../Tools.jl")

const global gilt_pars = Dict(
	"gilt_eps" => gilt_eps,
	"cg_chis" => collect(1:chi),
	"cg_eps" => cg_eps,
	"verbosity" => 3,
	"rotate" => rotate,
)


const global initialA_pars_low = Dict(
	"relT" => relT_low,
	"Jratio" => Jratio,
)

const global initialA_pars_high = Dict(
	"relT" => relT_high,
	"Jratio" => Jratio,
)

gilt_pars["verbosity"] = 0


@time search_result = perform_search(initialA_pars_low, initialA_pars_high, gilt_pars; search_tol = search_tol, max_number_of_steps = max_number_of_steps, verbose = true)

filename = "critical_temperatures/" * gilt_pars_identifier(gilt_pars) * "__tol=$(search_tol)"

# We only include Jratio in the file name if Jratio is not 1
# This insures backwards compatibility with earlier file names
if abs(Jratio - 1) > 1.e-10
	filename = filename * "_Jratio=$(Jratio)"
end

filename = filename * ".data"

println(Jratio, ":", (search_result[1]["relT"] + search_result[1]["relT"]) / 2)

serialize(filename, search_result)

if plot_trajectory
	gilt_pars["verbosity"] = 3

	# use average of low and high bounds on critical relT to plot the final trajectory
	initialA_pars = copy(search_result[1])
	initialA_pars["relT"] = (search_result[1]["relT"] + search_result[2]["relT"]) / 2

	plot_the_trajectory(initialA_pars, gilt_pars; traj_len = search_result[3])
end


