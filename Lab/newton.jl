# sections
# - FUNCTIONS
# - EXPERIMENT INIT
# - EIGENVALUES
# - NEWTON
# - SCALING DIMENSIONS
# - EXPORT

include("../Tools.jl");
include("../GaugeFixing.jl");
include("../KrylovTechnical.jl");

################################################
# section: FUNCTIONS
################################################

function gilt(A, pars)
    A = ju_to_py(A)
    A, log_fact, _ = py"gilttnr_step"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A, _ = fix_discrete_gauge(A)
    return py_to_ju(exp(log_fact) * A)
end

function gilt(A, list_of_elements, pars)
    A = ju_to_py(A)
    A, log_fact, _ = py"gilttnr_step"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A, _ = fix_discrete_gauge(A, list_of_elements)
    return py_to_ju(exp(log_fact) * A)
end


################################################
# section: EXPERIMENT INIT
################################################

chi = cla_or_def(1, 10)
gilt_eps = cla_or_def(2, 3e-4)
relT = cla_or_def(3, 0.9975949995592235)
number_of_initial_steps = cla_or_def(4, 20)
order = cla_or_def(5, 2)
stp = cla_or_def(6, 5e-6)
verbosity = cla_or_def(7, 0)

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

A_crit_approximation_JU = py_to_ju(A_crit_approximation);

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



function dgilt(δA)
    return df(x -> gilt(x, accepted_elements, gilt_pars), A_crit_approximation_JU, δA; stp=stp, order=order)
end

A_crit_approximation_JU /= (A_crit_approximation_JU |> norm);

A2 = gilt(A_crit_approximation_JU, accepted_elements, gilt_pars);

A_crit_approximation_JU |> norm
g = A2 |> norm;
N = g^(-1 / 3)

A_crit_approximation_JU = A_crit_approximation_JU * N;

A_crit_approximation_JU |> norm

A2 = gilt(A_crit_approximation_JU, accepted_elements, gilt_pars);


#################################################
# section: EIGENVALUES
#################################################


initial_vector = py_to_ju(random_Z2tens(A_crit_approximation));

res = eigsolve(dgilt, initial_vector, 10, :LM; verbosity=3, issymmetric=false, ishermitian=false, krylovdim=30);
res[1]



#################################################
# section: NEWTON
#################################################

function build_jacobian_approximation(vectors::Vector{t}, values::Vector) where {t}
    rank = length(values)
    jacobian_approximation = zeros(rank, rank)
    basis = Vector{t}(undef, rank)
    i = 1
    while i <= rank
        if imag(values[i]) == 0
            jacobian_approximation[i, i] = real(values[i])
            basis[i] = real(vectors[i])
            i += 1
        else
            if conj(values[i+1]) != values[i]
                throw("Unmatched complex eigenvlaue is detected")
            end
            λ₁ = real(values[i])
            λ₂ = imag(values[i])
            v1 = real(vectors[i])
            v2 = imag(vectors[i])
            v1norm = v1 |> norm
            v2norm = v2 |> norm
            v1 /= v1norm
            v2 /= v2norm
            jacobian_approximation[i, i] = λ₁
            jacobian_approximation[i+1, i+1] = λ₁
            jacobian_approximation[i+1, i] = -λ₂ * v2norm / v1norm
            jacobian_approximation[i, i+1] = λ₂ * v1norm / v2norm
            basis[i] = v1
            basis[i+1] = v2
            i += 2
        end
    end
    jacobian_approximation, basis
end


function build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis::Vector{t}) where {t}
    dim = length(non_orthogonal_normalised_basis)
    Graham_Schmidt_Matrix = zeros(dim, dim)
    Graham_Schmidt_Matrix[1, 1] = 1
    orthonormal_basis = t[non_orthogonal_normalised_basis[1]]
    for n = 1:(dim-1)
        new_vector = non_orthogonal_normalised_basis[n+1]
        for i = 1:n
            new_vector -= dot(orthonormal_basis[i], new_vector) * orthonormal_basis[i]
        end
        Nnp1 = norm(new_vector)
        new_vector /= Nnp1
        push!(orthonormal_basis, new_vector)

        gnn = Graham_Schmidt_Matrix[1:n, 1:n]
        v_old_dot_v_new_vector = zeros(n)
        for i = 1:n
            v_old_dot_v_new_vector[i] = dot(non_orthogonal_normalised_basis[i], non_orthogonal_normalised_basis[n+1])
        end
        Graham_Schmidt_Matrix[1:n, n+1] .= -1 / Nnp1 .* gnn * transpose(gnn) * v_old_dot_v_new_vector
        Graham_Schmidt_Matrix[n+1, n+1] = 1 / Nnp1
    end
    return Graham_Schmidt_Matrix, orthonormal_basis
end

jac_approximation_non_orthogonal_basis, non_orthogonal_normalised_basis = build_jacobian_approximation(res[2], res[1]);
g, orthonormal_basis = build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis);
jac_approximation = g^(-1) * jac_approximation_non_orthogonal_basis * g;

# check1
tst = transpose(g) * non_orthogonal_normalised_basis;
tst - orthonormal_basis .|> norm |> findmax

approximation_rank = length(orthonormal_basis);


#check2
ortho_check = zeros(approximation_rank, approximation_rank)
for i = 1:approximation_rank
    for j = 1:approximation_rank
        ortho_check[i, j] = dot(orthonormal_basis[i], orthonormal_basis[j])
    end
end
(ortho_check - diagm(diag(ortho_check))) |> norm
diag(ortho_check) .- 1 |> norm


function factor(δA)
    res = deepcopy(δA)
    for i = 1:approximation_rank
        for j = 1:approximation_rank
            res -= jac_approximation[i, j] * orthonormal_basis[i] * dot(orthonormal_basis[j], δA)
        end
    end
    return res
end

function newton_function(A)
    x_minus_f = A - gilt(A, accepted_elements, gilt_pars)
    initial_vector = py_to_ju(random_Z2tens(A_crit_approximation))
    correction = linsolve(factor, x_minus_f, initial_vector)[1]
    return A - correction
end


A_hist = [A_crit_approximation_JU];

N = 10
for i = 2:N
    push!(A_hist, newton_function(A_hist[i-1]))
    println(A_hist[i] - A_hist[i-1] |> norm)
end

mkpath("newton_results/chi=$chi")


fig = Figure(; size=(1000, 1000));

ax = Axis(fig[1, 1],
    title="Newton method convergence",
    titlesize=30,
    yscale=log10,
    xlabel="step",
    xlabelsize=35,
    xticklabelsize=20,
    ylabel="||Aₖ-Aₖ₊₁||",
    ylabelsize=35,
    yticklabelsize=20,);


lines!(ax, variation(A_hist) .|> norm);

fig

save("newton_results/chi=$chi/convergence.pdf", fig)


A_crit_approximation_new_JU = A_hist[end];

fig1 = plot_the_trajectory(
    ju_to_py(A_crit_approximation_JU),
    gilt_pars;
    traj_len=50
)

save("newton_results/chi=$chi/trajectory_starting_from_initial_tensor.pdf", fig1)


fig2 = plot_the_trajectory(
    ju_to_py(A_crit_approximation_new_JU),
    gilt_pars;
    traj_len=50
)

save("newton_results/chi=$chi/trajectory_starting_from_the_new_tensor.pdf", fig2)


#################################################
# section: SCALING DIMENSIONS
#################################################

py"""
def get_scaldims(A):
    logging.info("Diagonalizing the transfer matrix.")
    # The cost of this scales as O(chi^6).
    transmat = ncon((A, A), [[3,-101,4,-1], [4,-102,3,-2]])
    es = transmat.eig([0,1], [2,3], hermitian=False)[0]
    # Extract the scaling dimensions from the eigenvalues of the
    # transfer matrix.
    es = es.to_ndarray()
    es = np.abs(es)
    es = -np.sort(-es)
    es[es==0] += 1e-16  # Ugly workaround for taking the log of zero.
    log_es = np.log(es)
    log_es -= np.max(log_es)
    log_es /= -np.pi
    return log_es
"""

dims_old = py"get_scaldims"(ju_to_py(A_crit_approximation_JU));
dims_new = py"get_scaldims"(ju_to_py(A_crit_approximation_new_JU));

levels_found = length(dims_old)

low_levels = 2:20
levels = low_levels

fig = Figure(; size=(500, 500));
ax = Axis(fig[1, 1],
    title="Deviation of scaling dimensions from the exact values",
    ylabel="|Δ_approximate-Δ_exact| / |Δ_exact| "
)

nrms = (norm.(exact_spectrum(levels_found)[levels]))
lines!(ax, abs.(dims_old[levels] - exact_spectrum(levels_found)[levels]) ./ nrms, label="before newton")
lines!(ax, abs.(dims_new[levels] - exact_spectrum(levels_found)[levels]) ./ nrms, label="after newton")

axislegend(ax, position=:lt)

fig



#################################################
# section: EXPORT
#################################################

exprt = Dict(
    "initial tensor" => A_crit_approximation,
    "final tensor" => ju_to_py(A_crit_approximation_new_JU),
    "gilt_pars" => gilt_pars
)

serialize("newton_results/chi=$chi/newton_result.data", exprt)

#################################################
# section: SPECTRUM
#################################################

function dgilt_new(δA)
    return df(x -> gilt(x, accepted_elements, gilt_pars), A_crit_approximation_new_JU, δA; stp=stp, order=order)
end


initial_vector = py_to_ju(random_Z2tens(A_crit_approximation));

res_new = eigsolve(dgilt, initial_vector, 10, :LM; verbosity=3, issymmetric=false, ishermitian=false, krylovdim=30);
res_new[1]


#############################################################################
# section: CHECKING THE NUMBER OF BOND REPETITIONS AND THE RECURSION DEPTH
#############################################################################

A1, _ = py"gilttnr_step"(A_crit_approximation, 1.0, gilt_pars);

tmp = py"depth_dictionary"

recursion_depth = Dict(
    "S" => tmp[(1, "S")],
    "N" => tmp[(1, "N")],
    "E" => tmp[(1, "E")],
    "W" => tmp[(1, "W")]
)

serialize(flow_path * "/recursion_depth.data", recursion_depth)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth,
    "rotate" => rotate
)