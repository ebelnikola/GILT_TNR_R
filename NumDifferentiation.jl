# This module provides function df for computing numerical derivatives. 

module NumDifferentiation

export df

const global df_order_coefficients = Vector{Float64}[
	[0.0],
	[-1 / 2, 1 / 2],
	[1 / 12, -2 / 3, 2 / 3, -1 / 12],
	[-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60],
]

const global df_order_offsets = Array{Int64}[
	[0],
	[-1, 1],
	[-2, -1, 1, 2],
	[-3, -2, -1, 1, 2, 3],
]

"Returns the derivative of function f at the point x in the direction v. Parameters:
- stp::Float     : step size used to get the finite difference
- order = 2,3,4  : the order of the finite difference approximation (df(f,x,v;stp=stp, order=order)=vⁱ∂ᵢ f (x)+ O(step^order)) 
"
function df(f::Function, x, v; stp = 1e-8, order = 2)
	coefficients = df_order_coefficients[order]
	points = map(z -> z * stp * v + x, df_order_offsets[order])
	return sum(coefficients .* f.(points)) / stp
end



function df!(target, f::Function, x::Vector{t}, v::Vector{t}; stp = 1e-8, order = 2) where {t}
	coefficients = df_order_coefficients[order]
	offsets = df_order_offsets[order]
	for i ∈ eachindex(coefficients)
		target .+= f(x + offsets[i] * stp * v) * coefficients[i]
	end
	target ./= stp
end

"Returns Jacobian of the function f at the point x. Parameters:
- f::Function    : the function to be differentiated
- x::Vector      : the point at which the Jacobian is evaluated
- stp::Float     : step size used to get the finite difference
- order = 2,3,4  : the order of the finite difference approximation (df(f,x;stp=stp, order=order)=real jacobian + O(step^order)) 
- parallel::Bool : if true, the algorithm will compute derivatives in each direction in parallel. Note that this does not necessary boost the performance (E.g. if function is defined on a space of small dimension), check if it does before using. 
"
function df(f::Function, x::Vector{t}; stp = 1e-8, order = 2, parallel = false) where {t}
	dimDom = length(x)
	fx = f(x)
	dimCoDom = length(fx)

	M = zeros(eltype(fx), dimCoDom, dimDom)
	if parallel
		Threads.@threads for column ∈ 1:dimDom
			e = zeros(t, dimDom)
			e[column] = one(t)

			df!((@view M[:, column]), f, x, e; stp = stp, order = order)
		end
	else
		for column ∈ 1:dimDom
			e = zeros(t, dimDom)
			e[column] = one(t)

			df!((@view M[:, column]), f, x, e; stp = stp, order = order)
		end
	end

	return M
end

end

#=
module NumDifferrentiationTest

export run_test, tst_fun

using LinearAlgebra, Zygote, Main.NumDifferentiation

function set_signs(M; tol=1e-10)
	height, width = size(M)
	SM = diagm(sign.(M[1, :]) .^ (-1))
	return M * SM
end


function tst_fun(x)
	M = reshape(x, Int64(sqrt(length(x))), Int64(sqrt(length(x))))
	R = eigen(M + M').vectors
	set_signs(R)[:]
end

function run_test(stp, order, test_dim, n_tests)
	for _ = 1:n_tests
		A = 2 .* rand(test_dim) .- 1
		J0 = jacobian(tst_fun, A)[1]
		J2 = df(tst_fun, A; stp=stp, order=order)

		deviation = norm(J2 - J0) ./ norm(J0)

		println(deviation)
		if deviation > 10 * stp^(order)
			@warn "$deviation"
		end
	end
end
end
=#
#=
using .Differentiation, .DifferrentiationTest

function tst_fun2(x)
	M = reshape(x, Int64(sqrt(length(x))), Int64(sqrt(length(x))))
	R = eigen(M + M').vectors
	return R[:]
end

tst_dim = 4 * 4
A = 2 .* rand(tst_dim) .- 1
v = normalize!(2 .* rand(tst_dim) .- 1)
f(s) = df(tst_fun2, A, v; stp=s)

stps = logvec(-2, -0.1, -10)

difs = Vector{Vector{Float64}}(undef, length(stps))
for ind in eachindex(stps)
	difs[ind] = f(stps[ind])
end


fig = Figure(; resolution=(800, 800));
ax = Axis(fig[1, 1],
	title="Derivative covergence for x->eigen(x).vectors",
	#subtitle="chi=$chi; Derivative is computed using order $order algorithm",
	xlabel="step size",
	ylabel="relative change of the derivative",
	yscale=log10,
	xscale=log10,
	xreversed=true,
	xlabelsize=30,
	ylabelsize=30,
	xticklabelsize=30,
	yticklabelsize=30,
	titlesize=25,
	subtitlesize=25)
lines!(ax, stps[1:end-1], norm.(variation(difs)) ./ norm.(difs[1:end-1]))
fig

=#
