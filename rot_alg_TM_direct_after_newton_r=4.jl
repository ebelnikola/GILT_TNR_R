using Pkg
Pkg.activate(".")
include("Tools.jl")
include("KrylovTechnical.jl")
include("GaugeFixing.jl");



using TensorOperations
using KrylovKit



newton_res = deserialize("newton/rotate=true_30_6.0e-6_1.0e-10_jac_approximation_rank=54.data");
A = newton_res["A_newton"][15] |> ju_to_py;

gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:30),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "bond_repetitions" => 2,
    "recursion_depth" => newton_res["recursion_depth"],
    "rotate" => true,
)
Atmp, lf, _ = py"gilttnr_step"(A, 0.0, gilt_pars);
g = (Atmp.norm())^(-1 / 3)
A = A * g;
A = A.to_ndarray();


function parity(tens::Array{T,4}, Z) where {T}
    @tensor tens_flipped[-1, -2, -3, -4] := tens[1, 2, 3, 4] * Z[1, -1] * Z[2, -2] * Z[3, -3] * Z[4, -4]
    return dot(tens, tens_flipped) / (norm(tens))^2
end

Z = diagm(vcat(ones(15), -ones(15)));



function TM_direct_4(v; A=A)
    @tensoropt_verbose w[-1, -2, -3, -4] := A[5, -1, 6, 1] * A[6, -2, 7, 2] * A[7, -3, 8, 3] * A[8, -4, 5, 4] * v[1, 2, 3, 4]
end

vals, vecs = eigsolve(TM_direct_4, randn(30, 30, 30, 30), 50, :LM; krylovdim=60);

serialize("rot_alg_TM_direct_after_newton_r=4.data", (vals, vecs))


shifted_dimensions = -log.(vals) ./ pi * 2;

c = -shifted_dimensions[1] * 12;
dimensions_re = real.(shifted_dimensions .- shifted_dimensions[1]);
dimensions_im = imag.(shifted_dimensions .- shifted_dimensions[1]);


open("rot_alg_TM_direct_after_newton_r=4.log",
    "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            println("CENTRAL CHARGE: ", c)
            println("-------------------------------")
            println("SCALING DIMENSIONS (RE):")

            for i ∈ 2:50
                println(dimensions_re[i])
            end
            println("-------------------------------")

            println("SCALING DIMENSIONS (IM):")

            for i ∈ 2:50
                println(dimensions_im[i])
            end
            println("-------------------------------")

            println("Z2 EIGENVALUES:")

            for i ∈ 2:50
                println(real(parity(vecs[i], Z)))
            end
            println("-------------------------------")


            println("SPIN MOD 4")


            for i = 2:50
                v = vecs[i]
                @tensor spin = v[1, 2, 3, 4] * conj(v[2, 3, 4, 1])
                println(imag(log(spin) * 4 / (2 * π)))
            end

        end
    end
end