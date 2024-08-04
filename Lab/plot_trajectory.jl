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
    default = 1.0
    "--Jratio"
    help = "The anisotropy parm for initial A. 1 is isotropic value"
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
    "--rotate"
    help = "If true the algorithm will perform a rotation by 90 degrees after each GILT TNR step"
    arg_type = Bool
    default = false
    "--N"
    help = "The number of singular values of tensor to display in the trajectory plot"
    arg_type = Int64
    default = 20
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
    "verbosity" => 2,
    "rotate" => rotate
)


const global initialA_pars = Dict(
    "relT" => relT,
    "Jratio" => Jratio,
)

# if relT==0 then we search in critical_temperatures/ to find critical relT
search_tol=1.0e-10
if abs(relT)<1.e-10
    relT_low,relT_high,relT = find_critical_temperature(chi,gilt_pars,search_tol,Jratio)
end

initialA_pars["relT"] = relT

# delete this printing eventually
println("=========================================")
println("search for critical temp: chi=",chi," Jratio=",Jratio," found relT=",relT)
println("used gilt_pars=",gilt_pars)
println("=========================================")

plot_the_trajectory(initialA_pars, gilt_pars; traj_len=traj_len, N=N)


