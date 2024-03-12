# topics:
# - PYTHON MODULES
# - RANDOM TENSORS GENERATOR FOR TESTS AND OTHERS TEST FUNCTIONS
# - CONTINUOUS GAUGE
# - DISCRETE GAUGE



#############################################
# topic: PYTHON MODULES
#############################################

using PyCall, LinearAlgebra, Serialization
pushfirst!(pyimport("sys")."path", "GiltTNR")

py"""
import warnings
warnings.simplefilter('always')
"""

TENSORS = pyimport("tensors")
NCON = pyimport("ncon")
TENS = TENSORS.Tensor
TENSZ2 = TENSORS.TensorZ2
ncon = NCON.ncon




################################################
# topic: RANDOM TENSORS GENERATOR FOR TESTS AND OTHERS TEST FUNCTIONS
################################################


py"""

import numpy as np

def random_with_sign(*args, **kwargs):
    return 2*np.random.random_sample(*args,**kwargs)-1
"""

function random_Z2tens(A::PyObject)
    dims = A.shape
    qhape = A.qhape
    dirs = A.dirs
    t = TENSZ2.initialize_with(py"random_with_sign", dims, qhape=qhape, charge=0, invar=true, dirs=dirs)
    t = t / t.norm()
    return t
end

function random_Z2tens(chi::Int64)
    dims = fill(chi ÷ 2, 4, 2)
    qhape = hcat(zeros(Int64, 4), ones(Int64, 4))
    dirs = [-1, 1, 1, -1]
    t = TENSZ2.initialize_with(py"random_with_sign", dims, qhape=qhape, charge=0, invar=true, dirs=dirs)
    t = t / t.norm()
    return t
end

py"""
def check_sums_of_columns(U):
    Umat=U.to_ndarray()
    num_of_columns=Umat.shape[-1]
    Umat=np.reshape(Umat,(-1,num_of_columns))
    vec=np.apply_along_axis(np.sum,0,Umat)
    if vec[vec<0].shape[0]>0:
        warnings.warn("check sums of columns found a negative column sign!")
        return False
    if vec[vec<1e-7].shape[0]>0:
        warnings.warn("check sums of columns found out that there are columns with sums below 1e-7")
        return False
    return True
"""



#############################################
# topic: CONTINUOUS GAUGE
#############################################
function environment_for_vertical_gauge(tensors)
    connects = [[2, -1, 1, 3], [2, -2, 1, 3]]
    con_order = [2, 1, 3]
    environment = ncon(tensors, connects, con_order)
    environment /= environment.norm()
    return environment
end

function environment_for_vertical_gauge(tensors::Vector{Array{Float64,4}})
    connects = [[2, -1, 1, 3], [2, -2, 1, 3]]
    con_order = [2, 1, 3]
    environment = ncon(tensors, connects, con_order)
    normalize!(environment)
    return environment
end

function environment_for_horizontal_gauge(tensors)
    connects = [[-1, 2, 3, 1], [-2, 2, 3, 1]]
    con_order = [2, 3, 1]
    environment = ncon(tensors, connects, con_order)
    environment /= environment.norm()
    return environment
end

function environment_for_horizontal_gauge(tensors::Vector{Array{Float64,4}})
    connects = [[-1, 2, 3, 1], [-2, 2, 3, 1]]
    con_order = [2, 3, 1]
    environment = ncon(tensors, connects, con_order)
    normalize!(environment)
    return environment
end

function fix_continuous_gauge(A)
    tensors = [A, A.conj()]

    environment = environment_for_vertical_gauge(tensors)
    SV, V = environment.eig(0, 1, hermitian=true)
    A = ncon([A, V.conj(), V], [[-1, 2, -3, 4], [2, -2], [4, -4]])

    environment = environment_for_horizontal_gauge(tensors)
    SH, H = environment.eig(1, 0, hermitian=true) #wierdly, I needed to change the order of legs here to get corect directions in H. 
    A = ncon([A, H, H.conj()], [[1, -2, 3, -4], [1, -1], [3, -3]])
    return A, H, V, SH, SV
end



#=for _ = 1:1000
    chi = rand([10, 20, 30, 40])
    tst = random_Z2tens(chi)

    tst1, H1, V1, SH1, SV1 = fix_continuous_gauge(tst)
    tst2, H2, V2, SH2, SV2 = fix_continuous_gauge(tst1)

    c1 = (tst1 - tst2).norm() < 1e-10

    c2 = py"check_sums_of_columns"(H1)
    c3 = py"check_sums_of_columns"(V1)
    c4 = py"check_sums_of_columns"(H2)
    c5 = py"check_sums_of_columns"(V2)

    envh = environment_for_horizontal_gauge([tst1, tst1.conj()]).to_ndarray()
    c6 = (envh - diagm(diag(envh)) |> norm) < 1e-10

    envv = environment_for_vertical_gauge([tst1, tst1.conj()]).to_ndarray()
    c7 = (envv - diagm(diag(envv)) |> norm) < 1e-10

    if !(c1)
        @warn "problem"
    end

    if !(c2 && c3 && c4 && c5)
        @warn "problem"
    end

    if !(c6 && c7)
        @warn "problem"
    end
end=#


######################################################
# topic: DISCRETE GAUGE
######################################################

function flatten_the_vec_of_vec(vec)
    return reduce(vcat, vec)
end

function list_of_allowed_elements(A; tol=1e-7)
    number_of_threads = Threads.nthreads()
    element_lists = [Tuple{CartesianIndex{4},Float64}[] for _ in 1:number_of_threads]
    AA = A.to_ndarray()
    Threads.@threads for ind = CartesianIndices(size(AA))
        condition1 = abs(AA[ind]) > tol
        condition2 = !(ind[1] == ind[3] && ind[2] == ind[4])
        if condition1 && condition2
            push!(element_lists[Threads.threadid()], (ind, AA[ind]))
        end
    end
    return flatten_the_vec_of_vec(element_lists)
end


include("EchelonForm.jl")

function construct_a_raw_from_indices(ind, dimH, dimV)
    raw = zeros(Bool, dimH + dimV)
    raw[ind[1]] = !(raw[ind[1]])
    raw[ind[3]] = !(raw[ind[3]])
    raw[dimH+ind[2]] = !(raw[dimH+ind[2]])
    raw[dimH+ind[4]] = !(raw[dimH+ind[4]])
    return raw
end

function isindependent(index, preM, dimH, dimV)
    preM[end, :] = construct_a_raw_from_indices(index, dimH, dimV)
    _, r = echelon_form!(preM)
    if r == size(preM, 1)
        return true
    end
    return false
end

function construct_linear_system(list_of_elements, dimH, dimV)
    rows_num = dimH + dimV - 3
    preM = zeros(Bool, rows_num, dimH + dimV)
    preb = zeros(Bool, rows_num)
    accepted_elements = Vector{Tuple{CartesianIndex,Float64}}(undef, rows_num)
    cnt::Int64 = 1
    for el in list_of_elements
        if isindependent(el[1], preM[1:cnt, :], dimH, dimV)
            preM[cnt, :] .= construct_a_raw_from_indices(el[1], dimH, dimV)
            preb[cnt] = el[2] < 0
            accepted_elements[cnt] = el
            cnt += 1
        end

        if cnt > rows_num
            break
        end
    end
    if cnt > rows_num
        return preM, preb, accepted_elements
    else
        @warn "construct_linear_system: unable to find $rows_num independent rows, will fix only $(cnt-1) dofs"
        return preM[1:cnt-1, :], preb[1:cnt-1], accepted_elements[1:cnt-1]
    end
end

using AbstractAlgebra

const global ℤ₂ = GF(2)
const global _1m2 = one(ℤ₂)
const global _0m2 = zero(ℤ₂)


function to_bool(M)
    map(x -> x == _1m2 ? true : false, Matrix(M))
end

function solve_linear_system(preM, preb)
    𝕄 = matrix_space(ℤ₂, size(preM, 1), size(preM, 2))
    𝕍 = matrix_space(ℤ₂, size(preM, 1), 1)
    M = 𝕄(preM)
    b = 𝕍(preb)
    hv = solve(M, b)
    return to_bool(hv)
end

function bool_vec_to_HV(hv, dimH)
    H = map(x -> x ? -1.0 : 1.0, hv[1:dimH])
    V = map(x -> x ? -1.0 : 1.0, hv[dimH+1:end])
    return H, V
end

function get_HV_and_indices(A; tol=1e-7)
    # get the list of nonzero tensor elements
    l = list_of_allowed_elements(A; tol=tol)

    # order the list in the decreasing order
    sort!(l, by=x -> -abs(x[2]))
    dimH = sum(A.shape[1, :])
    dimV = sum(A.shape[1, :])

    preM, preb, accepted_elements = construct_linear_system(l, dimH, dimV)
    hv = solve_linear_system(preM, preb)
    H, V = bool_vec_to_HV(hv, dimH)

    return H, V, accepted_elements
end

function fix_discrete_gauge(A; tol=1e-7)
    H, V, accepted_elements = get_HV_and_indices(A; tol=tol)

    Hshape = A.shape[[1; 3], :]
    Hqhape = A.qhape[[1; 3], :]
    Hdirs = A.dirs[[1; 3]]
    HZ2 = TENSZ2.from_ndarray(diagm(H), shape=Hshape, qhape=Hqhape, charge=0, invar=true, dirs=Hdirs)


    Vshape = A.shape[[2; 4], :]
    Vqhape = A.qhape[[2; 4], :]
    Vdirs = A.dirs[[2; 4]]
    VZ2 = TENSZ2.from_ndarray(diagm(V), shape=Vshape, qhape=Vqhape, charge=0, invar=true, dirs=Vdirs)

    A = ncon([A, HZ2.conj(), VZ2.conj(), HZ2, VZ2], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    return A, accepted_elements, HZ2, VZ2
end

# If the list of independent elements is already known, we may speed up the construction
# of equation matrix dramaticaly by simply reusing the same list. However, since it was decided to store in the list both indices and values,
# it is important to update the values in the list for the new tensor.  

function new_list_of_elements(A, list_of_elements; tol=1e-7)
    AA = A.to_ndarray()
    list_of_elements_new = Vector{Tuple{CartesianIndex{4},Float64}}(undef, length(list_of_elements))
    for ind in eachindex(list_of_elements)
        new_entry = AA[list_of_elements[ind][1]]
        if abs(new_entry) < tol
            @warn "new_list_of_elements: new entry is below the threshold. It was $(list_of_elements[ind][2]) and became $(new_entry). Index $(list_of_elements[ind][1]) "
        end
        list_of_elements_new[ind] = (list_of_elements[ind][1], new_entry)
    end
    return list_of_elements_new
end

function get_HV_and_indices(A, list_of_elements; tol=1e-7)
    list_of_elements = new_list_of_elements(A, list_of_elements; tol=tol)
    dimH = sum(A.shape[1, :])
    dimV = sum(A.shape[1, :])

    preM, preb, accepted_elements = construct_linear_system(list_of_elements, dimH, dimV)
    hv = solve_linear_system(preM, preb)
    H, V = bool_vec_to_HV(hv, dimH)

    return H, V, accepted_elements
end

function fix_discrete_gauge(A, list_of_elements; tol=1e-7)
    H, V, accepted_elements = get_HV_and_indices(A, list_of_elements; tol=tol)

    Hshape = A.shape[[1; 3], :]
    Hqhape = A.qhape[[1; 3], :]
    Hdirs = A.dirs[[1; 3]]
    HZ2 = TENSZ2.from_ndarray(diagm(H), shape=Hshape, qhape=Hqhape, charge=0, invar=true, dirs=Hdirs)


    Vshape = A.shape[[2; 4], :]
    Vqhape = A.qhape[[2; 4], :]
    Vdirs = A.dirs[[2; 4]]
    VZ2 = TENSZ2.from_ndarray(diagm(V), shape=Vshape, qhape=Vqhape, charge=0, invar=true, dirs=Vdirs)

    A = ncon([A, HZ2.conj(), VZ2.conj(), HZ2, VZ2], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    return A, accepted_elements, HZ2, VZ2
end


#=for _ = 1:1000
    chi = rand([10, 20, 30, 40])
    tst1 = random_Z2tens(chi)
    tst2 = random_Z2tens(chi)

    tst1F, elements1, H1, V1 = fix_discrete_gauge(tst1)
    tst2F, elements2, H2, V2 = fix_discrete_gauge(tst2, elements1)

    tst1F = tst1F.to_ndarray()
    tst2F = tst2F.to_ndarray()

    c1 = true
    for el in elements1
        if tst1F[el[1]] <= 0
            c1 = false
            break
        end
    end

    c2 = true
    for el in elements2
        if tst2F[el[1]] <= 0
            c2 = false
            break
        end
    end



    if !(c1 && c2)
        @warn "problem"
    end

end=#

