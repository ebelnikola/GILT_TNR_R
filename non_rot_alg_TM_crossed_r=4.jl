using Pkg
Pkg.activate(".")
include("Tools.jl")
include("KrylovTechnical.jl")
include("GaugeFixing.jl");

using TensorOperations
using KrylovKit


traj = deserialize("trajectories/rotate=false_30_6.0e-6_1.0e-10__relT=1.0000110043212773_len=36.data");

A = traj["A"][25];

gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:30),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)
Atmp, lf, _ = py"gilttnr_step"(A, 0.0, gilt_pars);
g = (Atmp.norm())^(-1 / 3)
A = A * g;
A = A.to_ndarray();


function parity(tens::Array{T,5}, Z) where {T}
    @tensoropt tens_flipped[-1, -2, -3, -4, -5] := tens[1, 2, 3, 4, 5] * Z[1, -1] * Z[2, -2] * Z[3, -3] * Z[4, -4] * Z[5, -5]
    return dot(tens, tens_flipped) / (norm(tens))^2
end



Z = diagm(vcat(ones(15), -ones(15)));

function TM_crossed_4(v; A=A)
    @tensoropt_verbose w[-1, -2, -3, -4, -5] := A[-5, -1, 6, 1] * A[6, -2, 7, 2] * A[7, -3, 8, 3] * A[8, -4, 5, 4] * v[1, 2, 3, 4, 5]
end

vals, vecs = eigsolve(TM_crossed_4, randn(30, 30, 30, 30, 30), 50, :LM; krylovdim=60);

serialize("non_rot_alg_TM_crossed_r=4.data", (vals, vecs))


shifted_dimensions = -real.(log.(vals)) .* (17 / (8 * pi));

spins = imag.(log.((vals))) * (17 / (2 * pi));


c = -shifted_dimensions[1] * 12;

dimensions = shifted_dimensions .- shifted_dimensions[1];

open("non_rot_alg_TM_crossed_r=4.log",
    "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do


            println("CENTRAL CHARGE: ", c)
            println("-------------------------------")
            println("SCALING DIMENSIONS :")

            for i ∈ 2:50
                println(dimensions[i])
            end
            println("-------------------------------")
            println("SPINS mod 17:")

            for i ∈ 2:50
                println(spins[i])
            end

            println("-------------------------------")
            println("Z2 EIGENVALUES:")
            for i ∈ 2:50
                println(real(parity(vecs[i], Z)))
            end


        end
    end
end