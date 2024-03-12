include("../Tools.jl")


const global chi = cla_or_def(1, 10)
const global gilt_eps = cla_or_def(2, 5e-5)

const global relT = cla_or_def(3, 1.0)

const global traj_len = cla_or_def(4, 10)

const global N = cla_or_def(5, 20)

const global cg_eps = cla_or_def(6, 1e-10)
const global verbosity = cla_or_def(7, 3)


plot_the_trajectory(;
    chi=chi,
    relT=relT,
    traj_len=traj_len,
    N=N,
    gilt_eps=gilt_eps,
    cg_eps=cg_eps,
    verbosity=verbosity
)