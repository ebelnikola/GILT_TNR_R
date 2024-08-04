# topics:
# - PYTHON MODULES
# - RANDOM TENSORS GENERATOR FOR TESTS
# - Z2TENSOR JULIA STRUCTURE
# - KRILOVKIT MINIMAL VECTOR OPERATIONS

using KrylovKit, LinearAlgebra

################################################
# topic: PYTHON MODULES
################################################

using PyCall
pushfirst!(pyimport("sys")."path", "GiltTNR")
TENSORS = pyimport("tensors")
TENSZ2 = TENSORS.TensorZ2

################################################
# topic: RANDOM TENSORS GENERATOR FOR TESTS
################################################

py"""

import numpy as np


def random_with_sign(*args, **kwargs):
	return 2*np.random.random_sample(*args,**kwargs)-1
"""

"""
It could be confusing, but thi returns PyObject!
"""
function random_Z2tens(A::PyObject)
	dims = A.shape
	qhape = A.qhape
	dirs = A.dirs
	t = TENSZ2.initialize_with(py"random_with_sign", dims, qhape = qhape, charge = 0, invar = true, dirs = dirs)
	t = t / t.norm()
	return t
end

"""
It could be confusing, but thi returns PyObject!
"""
function random_Z2tens(chi::Int64)
	dims = fill(chi ÷ 2, 4, 2)
	qhape = hcat(zeros(Int64, 4), ones(Int64, 4))
	dirs = [-1, 1, 1, -1]
	t = TENSZ2.initialize_with(py"random_with_sign", dims, qhape = qhape, charge = 0, invar = true, dirs = dirs)
	t = t / t.norm()
	return t
end

################################################
# topic: Z2TENSOR JULIA STRUCTURE
################################################

# this structure is necessary for work with KrylovKit. Python types are not well supported and tend to throw the segmentation fault. 

struct Z2Tensor
	sects::Dict{NTuple{4, Int64}, Array}
	shape::Matrix{Int64}
	qhape::Matrix{Int64}
	dirs::Vector{Int64}
end

py"""
import copy
"""

function py_to_ju(A)
	return Z2Tensor(deepcopy(A.sects), deepcopy(A.shape), deepcopy(A.qhape), deepcopy(A.dirs))
end

function key_to_shape(shape, qhape, key, leg)
	index = findfirst(x -> x == key, qhape[leg, :])
	return shape[leg, index]
end

function extend_blocks_by_zeros(A::Z2Tensor, new_shape)
	extended_sections = Dict{NTuple{4, Int64}, Array}()
	for (key, block) in A.sects
		kts = leg -> key_to_shape(new_shape, A.qhape, key[leg], leg)
		chi1, chi2, chi3, chi4 = kts(1), kts(2), kts(3), kts(4)
		tmp = zeros(eltype(block), chi1, chi2, chi3, chi4)
		chi_old_1, chi_old_2, chi_old_3, chi_old_4 = size(block)
		tmp[1:chi_old_1, 1:chi_old_2, 1:chi_old_3, 1:chi_old_4] .= block
		extended_sections[key] = tmp
	end
	return Z2Tensor(extended_sections, new_shape, A.qhape, A.dirs)
end

function ju_to_py(A)
	out = TENSZ2(
		A.shape,
		qhape = A.qhape,
		dirs = A.dirs,
		invar = true,
		charge = 0,
	)
	py"""
	a=$out
	for k, v in $(A.sects).items():
		a[k]=copy.deepcopy(v)
	"""
	return out
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst = random_Z2tens(chi)
	tst2 = py_to_ju(tst)
	tst3 = ju_to_py(tst2)

	c1 = ((tst - tst3).norm() == 0)
	c2 = (tst.invar == tst3.invar)
	c3 = (tst.dirs == tst3.dirs)
	c4 = (tst.qhape == tst3.qhape)

	c5 = (tst.sects == tst2.sects)
	c6 = (tst.shape == tst2.shape)
	c7 = (tst.qhape == tst2.qhape)
	c8 = (tst.dirs == tst2.dirs)

	if !(c1 && c2 && c3 && c4)
		@warn "a problem"
	end

	if !(c5 && c6 && c7 && c8)
		@warn "a problem"
	end
end=#




import Base.getindex

function Base.getindex(A::Z2Tensor, ind...)
	return A.sects[ind]
end

function Base.getindex(A::Z2Tensor, ind)
	return A.sects[ind]
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst = py_to_ju(random_Z2tens(chi))
	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end
	c1 = tst[ind] == tst[ind...]
	c2 = tst[ind] == tst.sects[ind]

	if !(c1 && c2)
		@warn "problem"
	end
end=#


import Base.:+

function Base.:+(v::Z2Tensor, w::Z2Tensor)
	sum = deepcopy(v)
	for (k, block) in sum.sects
		block .+= w[k]
	end
	return sum
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])

	tst1 = py_to_ju(random_Z2tens(chi))
	tst2 = py_to_ju(random_Z2tens(chi))

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end

	tst3 = tst1 + tst2

	c1 = (tst3[ind] == (tst1[ind] + tst2[ind]))

	if !(c1)
		@warn "problem"
	end
end=#



import Base.:-

function Base.:-(v::Z2Tensor, w::Z2Tensor)
	diff = deepcopy(v)
	for (k, block) in diff.sects
		block .-= w[k]
	end
	return diff
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])

	tst1 = py_to_ju(random_Z2tens(chi))
	tst2 = py_to_ju(random_Z2tens(chi))

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end

	tst3 = tst1 - tst2

	c1 = (tst3[ind] == (tst1[ind] - tst2[ind]))

	if !(c1)
		@warn "problem"
	end
end=#



import Base.:/

function Base.:/(x::Z2Tensor, num)
	shape = x.shape
	qhape = x.qhape
	dirs = x.dirs
	sects = Dict{NTuple{4, Int64}, Array}()
	for (k, block) in x.sects
		sects[k] = block / num
	end
	return Z2Tensor(sects, shape, qhape, dirs)
end


#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])

	tst1 = py_to_ju(random_Z2tens(chi))
	num = 2 .* rand(ComplexF64) .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end

	tst2 = tst1 / num

	c1 = (tst2[ind] == tst1[ind] / num)

	if !(c1)
		@warn "problem"
	end
end=#


import Base.:*

function Base.:*(α, x::Z2Tensor)
	shape = x.shape
	qhape = x.qhape
	dirs = x.dirs
	sects = Dict{NTuple{4, Int64}, Array}()
	for (k, block) in x.sects
		sects[k] = block * α
	end
	return Z2Tensor(sects, shape, qhape, dirs)
end

function Base.:*(v::Z2Tensor, α)
	return Base.:*(α, v)
end


#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst1 = py_to_ju(random_Z2tens(chi))
	num = 2 .* rand(ComplexF64) .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end

	tst2 = tst1 * num

	c1 = (tst2[ind] == tst1[ind] * num)

	if !(c1)
		@warn "problem"
	end
end
=#

import Base.real, Base.imag

function real(x::Z2Tensor)
	shape = x.shape
	qhape = x.qhape
	dirs = x.dirs
	sects = Dict{NTuple{4, Int64}, Array}()
	for (k, block) in x.sects
		sects[k] = real.(block)
	end
	return Z2Tensor(sects, shape, qhape, dirs)
end

function imag(x::Z2Tensor)
	shape = x.shape
	qhape = x.qhape
	dirs = x.dirs
	sects = Dict{NTuple{4, Int64}, Array}()
	for (k, block) in x.sects
		sects[k] = imag.(block)
	end
	return Z2Tensor(sects, shape, qhape, dirs)
end


#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst1 = py_to_ju(random_Z2tens(chi))
	num = 2 .* rand(ComplexF64) .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end

	tst1 = tst1 * num

	re = real(tst1)
	img = imag(tst1)

	c1 = (re[ind] == real.(tst1[ind]))

	c2 = (img[ind] == imag.(tst1[ind]))

	if !(c1 && c2)
		@warn "problem"
	end
end=#


################################################
# topic: KRILOVKIT MINIMAL VECTOR OPERATIONS 
################################################
# There is a list of functions which should work with python tensors id we want to use KrylovKit

##### Base.:*(α, v): multiply v with a scalar α, which can be of a different scalar type; in particular this method is used to create vectors similar to v but with a different type of underlying scalars. 



# done above



##### Base.similar(v): a way to construct vectors which are exactly similar to v

import Base.similar
function Base.similar(x::Z2Tensor)
	shape = x.shape
	qhape = x.qhape
	dirs = x.dirs
	sects = Dict{NTuple{4, Int64}, Array}()
	for (k, block) in x.sects
		sects[k] = similar(block)
	end
	return Z2Tensor(sects, shape, qhape, dirs)
end



import Base.zero
function Base.zero(x::Z2Tensor)
	shape = x.shape
	qhape = x.qhape
	dirs = x.dirs
	sects = Dict{NTuple{4, Int64}, Array}()
	for (k, block) in x.sects
		sects[k] = zero(block)
	end
	return Z2Tensor(sects, shape, qhape, dirs)
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])

	tst1 = py_to_ju(random_Z2tens(chi))

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end

	tst2 = similar(tst1)

	c1 = (tst2.dirs == tst1.dirs)
	c2 = (tst2.qhape == tst1.qhape)
	c3 = (tst2.shape == tst1.shape)




	if !(c1 && c2 && c3)
		@warn "problem"
	end
end=#




##### LinearAlgebra.mul!(w, v, α): out of place scalar multiplication; multiply vector v with scalar α and store the result in w

import LinearAlgebra.mul!

function LinearAlgebra.mul!(w::Z2Tensor, v::Z2Tensor, α)
	for (k, block) in w.sects
		block .= v.sects[k] * α
	end
	return w
end


#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst1 = py_to_ju(random_Z2tens(chi))
	tst2 = similar(tst1)
	num = 2 .* rand() .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end


	mul!(tst2, tst1, num)

	c1 = (tst2[ind] == tst1[ind] * num)



	if !(c1)
		@warn "problem"
	end
end=#




##### LinearAlgebra.rmul!(v, α): in-place scalar multiplication of v with α; in particular with α = false, v is the corresponding zero vector

import LinearAlgebra.rmul!


function LinearAlgebra.rmul!(v::Z2Tensor, α)
	for (_, block) in v.sects
		block .*= α
	end
	return v
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])

	tst1 = py_to_ju(random_Z2tens(chi))
	tst2 = deepcopy(tst1)
	num = 2 .* rand() .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end


	rmul!(tst1, num)

	c1 = (tst1[ind] == tst2[ind] * num)



	if !(c1)
		@warn "problem"
	end
end=#



##### LinearAlgebra.axpy!(α, v, w): store in w the result of α*v + w

import LinearAlgebra.axpy!

function LinearAlgebra.axpy!(α, v::Z2Tensor, w::Z2Tensor)
	for (k, block) in w.sects
		block .+= α * v.sects[k]
	end
	return w
end


#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst1 = py_to_ju(random_Z2tens(chi))
	tst2 = py_to_ju(random_Z2tens(chi))
	tst2c = deepcopy(tst2)
	num = 2 .* rand() .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end


	axpy!(num, tst1, tst2)


	c1 = (tst2[ind] == tst1[ind] * num + tst2c[ind])



	if !(c1)
		@warn "problem"
	end
end=#


##### LinearAlgebra.axpby!(α, v, β, w): store in w the result of α*v + β*w

import LinearAlgebra.axpby!

function LinearAlgebra.axpby!(α, v::Z2Tensor, β, w::Z2Tensor)
	for (k, block) in w.sects
		block .= α * v.sects[k] + β * block
	end
	return w
end

#=for _ = 1:1000
	chi = rand([10, 20, 30, 40])
	tst1 = py_to_ju(random_Z2tens(chi))
	tst2 = py_to_ju(random_Z2tens(chi))
	tst2c = deepcopy(tst2)
	num1 = 2 .* rand() .- 1
	num2 = 2 .* rand() .- 1

	ind = Tuple(rand([0, 1], 4))
	while sum(ind) % 2 != 0
		ind = Tuple(rand([0, 1], 4))
	end


	axpby!(num1, tst1, num2, tst2)


	c1 = (tst2[ind] == tst1[ind] * num1 + tst2c[ind] * num2)



	if !(c1)
		@warn "problem"
	end
end=#



##### LinearAlgebra.dot(v,w): compute the inner product of two vectors

import LinearAlgebra.dot
function LinearAlgebra.dot(v::Z2Tensor, w::Z2Tensor)
	res = 0.0
	for (k, _) in v.sects
		res += dot(v[k], w[k])
	end
	return res
end





##### LinearAlgebra.norm(v): compute the 2-norm of a vector
import LinearAlgebra.norm
function LinearAlgebra.norm(v::Z2Tensor)
	nrmsq = 0.0
	for (_, block) in v.sects
		nrmsq += norm(block)^2
	end
	return sqrt(nrmsq)
end

