include("../Tools.jl")


settings = ArgParseSettings()
@add_arg_table! settings begin
    "--chi"
    help = "The bond dimension"
    arg_type = Int64
    default = 10
    "--gilt_eps"
    help = "The threshold used in the GILT algorithm"
    arg_type = Float64
    default = 3e-4
    "--relT"
    help = "The realtive temperature of the initial tensor"
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



const global gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 2,
    "rotate" => rotate
)

plot_the_trajectory(relT, gilt_pars; traj_len=traj_len, N=N)


