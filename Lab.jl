include("Tools.jl");
include("GaugeFixing.jl");
include("KrylovTechnical.jl")



################################################
# topic: EXPERIMENT INIT
################################################
chi = cla_or_def(1, 30)
gilt_eps = cla_or_def(2, 5e-6)
relT = cla_or_def(3, 1.000009865760803)
number_of_initial_steps = cla_or_def(4, 15)
verbosity = cla_or_def(6, 0)
bond_repetitions = cla_or_def(7, 2)
recursion_depth = cla_or_def(8, 11)
cg_eps = cla_or_def(7, 1e-10)

const global gilt_pars_local = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => verbosity,
    "bond_repetitions" => bond_repetitions,
    "recursion_depth" => recursion_depth,
)

A_crit_approximation_tmp = trajectory(relT, number_of_initial_steps, gilt_pars_local)["A"][end];
A_crit_approximation_tmp, _ = fix_continuous_gauge(A_crit_approximation_tmp);
A_crit_approximation_tmp = A_crit_approximation_tmp / A_crit_approximation_tmp.norm()
const global A_crit_approximation, list_of_elements = fix_discrete_gauge(A_crit_approximation_tmp; tol=1e-7);

const global A_crit_approximation_JU = py_to_ju(A_crit_approximation)

A_crit_approximation_tmp = nothing;



################################################
# topic: gilts derivative 
################################################

function gilt(A, list_of_elements, pars)
    A = ju_to_py(A)
    A, _ = py"gilttnr_step"(A, 1.0, pars)
    A, _ = fix_continuous_gauge(A)
    A, _ = fix_discrete_gauge(A, list_of_elements; tol=1e-7)
    A = A / A.norm()
    return py_to_ju(A)
end

function dgilt(δA)
    return df(x -> gilt(x, list_of_elements, gilt_pars_local), A_crit_approximation_JU, δA; stp=1e-7, order=2)
end



################################################
# topic: Generation of random tensors
################################################

py"""
def random_with_sign(*args, **kwargs):
    return 2*np.random.random_sample(*args,**kwargs)-1
"""

function random_Z2tens(A)
    dims = A.shape
    qhape = A.qhape
    dirs = A.dirs
    t = TENSZ2.initialize_with(py"random_with_sign", dims, qhape=qhape, charge=0, invar=true, dirs=dirs)
    t = t / t.norm()
    return t
end

################################################
# topic: Eigenvalues computation
################################################
initial_vector = py_to_ju(random_Z2tens(A_crit_approximation));

res = eigsolve(dgilt, initial_vector, 3; verbosity=3, issymmetric=false, ishermitian=false);

v1 = real(res[2][2])
v2 = real(res[2][3])
v2 = v2 - dot(v1, v2) * v1
v2 /= norm(v2)


dgilt_no_v1v2 = x -> dgilt(x) - dot(x, v1) * v1 - dot(x, v2) * v2;
res = eigsolve(dgilt_no_v1v2, initial_vector, 3; verbosity=3, issymmetric=false, ishermitian=false);














##########################
# topic: scaling dimensions from tranfer matrix. 
##########################
using TensorOperations

tst = A_crit_approximation.to_ndarray();
@tensor transfer[i1, i2, j1, j2] := tst[1, i1, 2, j1] * tst[2, i2, 1, j2];
transfer = reshape(transfer, 900, 900);

vls = abs.(eigen(transfer, sortby=x -> -abs(x)).values)

sp = -log.(vls)
sp .-= sp[1]

sp ./= (sp[2] / 0.125)

sp[1:10]
