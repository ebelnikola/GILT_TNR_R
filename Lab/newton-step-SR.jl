# sections
# - FUNCTIONS
# - NEWTON
# - EIGENVALUES


################################################
# section: FUNCTIONS
################################################


include("../Tools.jl");
include("../GaugeFixing.jl");
include("../KrylovTechnical.jl");


function embedded_distance(tens1, tens2)
	if tens1.qhape != tens2.qhape
		throw("nonequal qhapes, code is not suited for such a situation.")
	end
	extended_shape = max.(tens1.shape, tens2.shape)
	tens1_ext = extend_blocks_by_zeros(tens1, extended_shape)
	tens2_ext = extend_blocks_by_zeros(tens2, extended_shape)
	tens1_ext - tens2_ext |> norm
end

function truncated_distance_with_additional_sign_fixing(tens1, tens2)
	if tens1.qhape != tens2.qhape
		throw("nonequal qhapes, code is not suited for such a situation.")
	end
	truncated_shape = min.(tens1.shape, tens2.shape)
	tens1_tr = (truncate_blocks(tens1, truncated_shape) |> ju_to_py)
	tens2_tr = (truncate_blocks(tens2, truncated_shape) |> ju_to_py)
	tens2_f, tens2_accepted_elements, _ = fix_discrete_gauge(tens2_tr)
	tens1_f, _ = fix_discrete_gauge(tens1_tr, tens2_accepted_elements)
	tens1_f.to_ndarray() - tens2_f.to_ndarray() |> norm
end

function embedded_distance_with_additional_sign_fixing(tens1, tens2)
	if tens1.qhape != tens2.qhape
		throw("nonequal qhapes, code is not suited for such a situation.")
	end
	extended_shape = max.(tens1.shape, tens2.shape)
	tens1_ext = (extend_blocks_by_zeros(tens1, extended_shape) |> ju_to_py)
	tens2_ext = (extend_blocks_by_zeros(tens2, extended_shape) |> ju_to_py)
	tens2_f, tens2_accepted_elements, _ = fix_discrete_gauge(tens2_ext)
	tens1_f, _ = fix_discrete_gauge(tens1_ext, tens2_accepted_elements)
	tens1_f.to_ndarray() - tens2_f.to_ndarray() |> norm
end;


function gilt(A, pars)
	A = ju_to_py(A)
	A, _ = py"gilttnr_step"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A)
	A /= A.norm()
	return py_to_ju(A)
end

function gilt(A, list_of_elements, pars; trunc_shape = nothing)
	A = ju_to_py(A)
	A, _ = py"gilttnr_step"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A = py_to_ju(A)
	if !isnothing(trunc_shape)
		A = truncate_blocks(A, trunc_shape)
	end
	A, _ = fix_discrete_gauge(ju_to_py(A), list_of_elements)
	A /= A.norm()
	Aju = py_to_ju(A)
	return Aju
end


function fix_discrete_gauge(A::Z2Tensor; tol = 1e-7)
    A1py, accepted_elements, _ = fix_discrete_gauge(ju_to_py(A); tol = tol)
    return py_to_ju(A1py), accepted_elements
end

function fp_error_with_shape(A::Z2Tensor, accepted_elements, pars; trunc_shape = nothing)
    RA = gilt(A, accepted_elements, pars; trunc_shape = trunc_shape)        
    return embedded_distance(RA, A), RA.shape
end

#################################################
# section: NEWTON
#################################################

function jacobian_eigsystem(A, N, list_of_elements, gilt_pars; trunc_shape = nothing)
    function dgilt(δA)
        return df(x -> gilt(x, list_of_elements, gilt_pars; trunc_shape = trunc_shape), A, δA; stp = 1e-4, order = 2)
    end
    initial_vector = py_to_ju(random_Z2tens(ju_to_py(A)));
    res = eigsolve(dgilt, initial_vector, N, :LM; verbosity = 1, issymmetric = false, ishermitian = false, krylovdim = N + 20, maxiter = 200)
    return res
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
	for n ∈ 1:(dim-1)
		new_vector = non_orthogonal_normalised_basis[n+1]
		for i ∈ 1:n
			new_vector -= dot(orthonormal_basis[i], new_vector) * orthonormal_basis[i]
		end
		Nnp1 = norm(new_vector)
		new_vector /= Nnp1
		push!(orthonormal_basis, new_vector)

		gnn = Graham_Schmidt_Matrix[1:n, 1:n]
		v_old_dot_v_new_vector = zeros(n)
		for i ∈ 1:n
			v_old_dot_v_new_vector[i] = dot(non_orthogonal_normalised_basis[i], non_orthogonal_normalised_basis[n+1])
		end
		Graham_Schmidt_Matrix[1:n, n+1] .= -1 / Nnp1 .* gnn * transpose(gnn) * v_old_dot_v_new_vector
		Graham_Schmidt_Matrix[n+1, n+1] = 1 / Nnp1
	end
	return Graham_Schmidt_Matrix, orthonormal_basis
end

function newton_correction(A, eigensystem_size_for_jacobian, list_of_elements, gilt_pars; trunc_shape = nothing)
    
    # compute eigensystem of jacobian
    initial_vector = py_to_ju(random_Z2tens(ju_to_py(A)))
    eigensystem_init = jacobian_eigsystem(A, eigensystem_size_for_jacobian, list_of_elements, gilt_pars; trunc_shape = trunc_shape) 
    println("EIGENVALUES (INITIAL):")
    for val in eigensystem_init[1]
        println(val)
    end  

    if length(eigensystem_init[1]) > eigensystem_size_for_jacobian
        if conj(eigensystem_init[1][eigensystem_size_for_jacobian]) ≈ eigensystem_init[1][eigensystem_size_for_jacobian+1]
            approximation_rank = eigensystem_size_for_jacobian + 1
        else
            approximation_rank = eigensystem_size_for_jacobian
        end
    else
        approximation_rank = eigensystem_size_for_jacobian
    end

    eigensystem_init = [eigensystem_init[1][1:approximation_rank], eigensystem_init[2][1:approximation_rank]];

    # compute approxiate jacobian

    jac_approximation_non_orthogonal_basis, non_orthogonal_normalised_basis = build_jacobian_approximation(eigensystem_init[2], eigensystem_init[1]);

    Graham_Schmidt_matrix, orthonormal_basis = build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis);

    jac_approximation = Graham_Schmidt_matrix^(-1) * jac_approximation_non_orthogonal_basis * Graham_Schmidt_matrix;

    ImJ_inv_matrix = (I - jac_approximation)^(-1);

    function project_to_Vs(δA)
        res = zero(δA)
        for i in 1:approximation_rank
            res += orthonormal_basis[i] * dot(orthonormal_basis[i], δA)
        end
        return res
    end

    function ImJ_inv(δA)
        δA_in_Vs = project_to_Vs(δA)
        ImJ_inv_δA_in_Vs = zero(δA)
        δA_out_of_Vs = δA - δA_in_Vs
        for i ∈ 1:approximation_rank
            for j ∈ 1:approximation_rank
                ImJ_inv_δA_in_Vs += ImJ_inv_matrix[i, j] * orthonormal_basis[i] * dot(orthonormal_basis[j], δA)
            end
        end
        return ImJ_inv_δA_in_Vs + δA_out_of_Vs
    end

    # compute correction and return it

	x_minus_f = A - gilt(A, list_of_elements, gilt_pars; trunc_shape = trunc_shape)
	correction = -1.0 * ImJ_inv(x_minus_f)
	return correction
end

function newton_correction_with_iterations_fixed(A, eigensystem_size_for_jacobian, list_of_elements, gilt_pars; trunc_shape = nothing)

    # compute eigensystem of jacobian

	A1, _ = py"gilttnr_step"(ju_to_py(A), 0.0, gilt_pars);

	tmp = py"depth_dictionary"
	println(tmp)
	flush(stdout)

	recursion_depth = Dict(
		"S" => tmp[(1, "S")],
		"N" => tmp[(1, "N")],
		"E" => tmp[(1, "E")],
		"W" => tmp[(1, "W")],
	)

	gilt_pars1 = deepcopy(gilt_pars)

	gilt_pars1["bond_repetitions"] = 2
	gilt_pars1["recursion_depth"] = recursion_depth

    initial_vector = py_to_ju(random_Z2tens(ju_to_py(A)))
    eigensystem_init = jacobian_eigsystem(A, eigensystem_size_for_jacobian, list_of_elements, gilt_pars1; trunc_shape = trunc_shape) 
    println("EIGENVALUES (INITIAL):")
    for val in eigensystem_init[1]
        println(val)
    end  

    if length(eigensystem_init[1]) > eigensystem_size_for_jacobian
        if conj(eigensystem_init[1][eigensystem_size_for_jacobian]) ≈ eigensystem_init[1][eigensystem_size_for_jacobian+1]
            approximation_rank = eigensystem_size_for_jacobian + 1
        else
            approximation_rank = eigensystem_size_for_jacobian
        end
    else
        approximation_rank = eigensystem_size_for_jacobian
    end

    eigensystem_init = [eigensystem_init[1][1:approximation_rank], eigensystem_init[2][1:approximation_rank]];

    # compute approxiate jacobian

    jac_approximation_non_orthogonal_basis, non_orthogonal_normalised_basis = build_jacobian_approximation(eigensystem_init[2], eigensystem_init[1]);

    Graham_Schmidt_matrix, orthonormal_basis = build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis);

    jac_approximation = Graham_Schmidt_matrix^(-1) * jac_approximation_non_orthogonal_basis * Graham_Schmidt_matrix;

    ImJ_inv_matrix = (I - jac_approximation)^(-1);

    function project_to_Vs(δA)
        res = zero(δA)
        for i in 1:approximation_rank
            res += orthonormal_basis[i] * dot(orthonormal_basis[i], δA)
        end
        return res
    end

    function ImJ_inv(δA)
        δA_in_Vs = project_to_Vs(δA)
        ImJ_inv_δA_in_Vs = zero(δA)
        δA_out_of_Vs = δA - δA_in_Vs
        for i ∈ 1:approximation_rank
            for j ∈ 1:approximation_rank
                ImJ_inv_δA_in_Vs += ImJ_inv_matrix[i, j] * orthonormal_basis[i] * dot(orthonormal_basis[j], δA)
            end
        end
        return ImJ_inv_δA_in_Vs + δA_out_of_Vs
    end

    # compute correction and return it

	x_minus_f = A - gilt(A, list_of_elements, gilt_pars1; trunc_shape = trunc_shape)
	correction = -1.0 * ImJ_inv(x_minus_f)
	return correction
end





