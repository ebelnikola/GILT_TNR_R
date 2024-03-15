# sections
# - EXPERIMENT INIT
# - RECURSION DEPTH FIXING
# - FUNCTIONS
# - NEWTON
# - EIGENVALUES


################################################
# section: EXPERIMENT INIT
################################################

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
    "--number_of_initial_steps"
    help = "Number of RG steps made to get an approximation of the critical tensor"
    arg_type = Int64
    default = 5
    "--cg_eps"
    help = "The threshold used in TRG steps to truncate the bonds"
    arg_type = Float64
    default = 1e-10
    "--ord"
    help = "order of differentiation algorithm"
    arg_type = Int64
    default = 2
    "--stp"
    help = "step used in differentiation algorithm"
    arg_type = Float64
    default = 5e-6
    "--eigensystem_size"
    help = "Number of eigenvectors and eigenvalues ot be used to approximate the jacobian"
    arg_type = Int64
    default = 10
    "--N"
    help = "Number of steps performed by the Newton algorithm"
    arg_type = Int64
    default = 20
    "--verbosity"
    help = "Verbosity of the eigensolver"
    arg_type = Int64
    default = 3
end

rotate = true

pars = parse_args(settings; as_symbols=true)
for (key, value) in pars
    @eval $key = $value
end

include("../Tools.jl");
include("../GaugeFixing.jl");
include("../KrylovTechnical.jl");



gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate
)

A_crit_approximation = trajectory(relT, number_of_initial_steps, gilt_pars)["A"][end];
A_crit_approximation, Hc, Vc, SHc, SVc = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements = fix_discrete_gauge(A_crit_approximation; tol=1e-7);

A_crit_approximation /= A_crit_approximation.norm();



################################################
# section: RECURSION DEPTH FIXING AND NORMALISATION
################################################

A1, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars);

g = A1.norm();
A_crit_approximation = g^(-1 / 3) * A_crit_approximation;
A_crit_approximation_JU = py_to_ju(A_crit_approximation);


tmp = py"depth_dictionary"

recursion_depth = Dict(
    "S" => tmp[(1, "S")],
    "N" => tmp[(1, "N")],
    "E" => tmp[(1, "E")],
    "W" => tmp[(1, "W")]
)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth,
    "rotate" => rotate
)

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


function dgilt(δA)
    return df(x -> gilt(x, accepted_elements, gilt_pars), A_crit_approximation_JU, δA; stp=stp, order=ord)
end


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




#################################################
# section: NEWTON
#################################################


initial_vector = py_to_ju(random_Z2tens(A_crit_approximation));
eigensystem_init = eigsolve(dgilt, initial_vector, eigensystem_size, :LM; verbosity=verbosity, issymmetric=false, ishermitian=false, krylovdim=30);


jac_approximation_non_orthogonal_basis, non_orthogonal_normalised_basis = build_jacobian_approximation(eigensystem_init[2], eigensystem_init[1]);
Graham_Schmidt_matrix, orthonormal_basis = build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis);
jac_approximation = Graham_Schmidt_matrix^(-1) * jac_approximation_non_orthogonal_basis * Graham_Schmidt_matrix;

approximation_rank = length(eigensystem_init[1])

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

for i = 2:N
    push!(A_hist, newton_function(A_hist[i-1]))
    println("Newton step size: $(A_hist[i] - A_hist[i-1] |> norm)")
end

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


save("newton/" * gilt_pars_identifier(gilt_pars) * "__newton_convergence_plot.pdf", fig)


A_crit_approximation_new_JU = A_hist[end];

fig = plot_the_trajectory(
    ju_to_py(A_crit_approximation_new_JU),
    gilt_pars;
    traj_len=50
)
save("newton/" * gilt_pars_identifier(gilt_pars) * "__trajectory_after_newton.pdf", fig)

#################################################
# section: EIGENVALUES
#################################################

function dgilt_after_newton(δA)
    return df(x -> gilt(x, accepted_elements, gilt_pars), A_crit_approximation_new_JU, δA; stp=stp, order=ord)
end


initial_vector = py_to_ju(random_Z2tens(A_crit_approximation));
eigensystem = eigsolve(dgilt_after_newton, initial_vector, eigensystem_size, :LM; verbosity=verbosity, issymmetric=false, ishermitian=false, krylovdim=30);


result = Dict(
    "A" => ju_to_py(A_crit_approximation_new_JU),
    "A_init" => A_crit_approximation,
    "eigensystem" => eigensystem,
    "eigensystem_init" => eigensystem_init,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth
)

serialize("newton/" * gilt_pars_identifier(gilt_pars) * "__newton_result.data", result)

println("EIGENVALUES:")
for val in eigensystem[1]
    println(val)
end



