# sections
# - EXPERIMENT INIT 
# - APPROPRIATE NORMALISATION OF A_crit_approximation
# - RETRIEVING GAUGE TRANSFORMATIONS AND R MATRICES
# - DEFINING N,S,W,E PIECES
# - DEFINING A1,A2
# - DEFINING t1_1,t1_2,t2_1,t2_2
# - DEFINING R,D,L,U
# - DEFINING hL, hR, vB and vT - INDEX COMPRESSION MATRICES
# - DEFINING THE SCALING OPERATOR
# - COMPUTING THE EIGENVALUES

################################################
# section: EXPERIMENT INIT
################################################

using ArgParse

settings = ArgParseSettings()
@add_arg_table! settings begin
	"--chi"
	help = "The bond dimension"
	arg_type = Int64
	default = 30
	"--gilt_eps"
	help = "The threshold used in the GILT algorithm"
	arg_type = Float64
	default = 6e-6
	"--relT"
	help = "The relative temperature of the initial tensor"
	arg_type = Float64
	default = 1.0000110043212773
	"--Jratio"
	help = "The anisotropy parameter of the initial tensor"
	arg_type = Float64
	default = 1.0
	"--rotate"
	help = "If true the algorithm will perform a rotation by 90 degrees after each GILT TNR step"
	arg_type = Bool
	default = false
	"--number_of_initial_steps"
	help = "Number of RG steps made to get an approximation of the critical tensor"
	arg_type = Int64
	default = 23
	"--cg_eps"
	help = "The threshold used in TRG steps to truncate the bonds"
	arg_type = Float64
	default = 1e-10
	"--N"
	help = "Number of eigenvalues to compute"
	arg_type = Int64
	default = 10
	"--verbosity"
	help = "Verbosity of the eigensolver"
	arg_type = Int64
	default = 0
	"--krylovdim"
	help = "Dimension of the Krylov space used in the eigenvalue computation. This parameter should be larger than N."
	arg_type = Int64
	default = 30
	"--chi_compression"
	help = "The bond dimension used to compress 8 leg tensors down to 4 leg tensors"
	arg_type = Int64
	default = 20
	"--path_to_tensor"
	help = "If provided, the code will ignore relT, Jratio, and number_of_initial_steps and will load the initial tensor from the given path. The file with the tensor should be created using Julia's serialize function. It should be a dictionary that contains the tensor A under key \"A\" and the recursion depths array under key \"recursion_depth\"."
	arg_type = String
	default = "none"
end

pars = parse_args(settings; as_symbols = true)
for (key, value) in pars
	@eval $key = $value
end

include("../Tools.jl");
include("../GaugeFixing.jl");
include("../KrylovTechnical.jl");

if path_to_tensor == "none"

	gilt_pars = Dict(
		"gilt_eps" => gilt_eps,
		"cg_chis" => collect(1:chi),
		"cg_eps" => cg_eps,
		"verbosity" => 0,
		"rotate" => rotate,
	)
	initialA_pars = Dict("relT" => relT, "Jratio" => Jratio)
	A_crit_approximation = trajectory(initialA_pars, number_of_initial_steps, gilt_pars)["A"][end]

	A_crit_approximation, _ = fix_continuous_gauge(A_crit_approximation)
	A_crit_approximation, accepted_elements, _ = fix_discrete_gauge(A_crit_approximation; tol = 1e-7)
	A_crit_approximation /= A_crit_approximation.norm()

else
	dt = deserialize(path_to_tensor)
	A_crit_approximation = dt["A"]
	recursion_depth = dt["recursion_depth"]


	gilt_pars = Dict(
		"gilt_eps" => gilt_eps,
		"cg_chis" => collect(1:chi),
		"cg_eps" => cg_eps,
		"verbosity" => 0,
		"bond_repetitions" => 2,
		"recursion_depth" => recursion_depth,
		"rotate" => rotate,
	)

	A_crit_approximation, _ = fix_continuous_gauge(A_crit_approximation)
	A_crit_approximation, accepted_elements, _ = fix_discrete_gauge(A_crit_approximation; tol = 1e-7)
	A_crit_approximation /= A_crit_approximation.norm()
end

##############################################################
# section: APPROPRIATE NORMALISATION OF A_crit_approximation
##############################################################

Atmp, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars);

g = (Atmp.norm())^(-1 / 3)
A_crit_approximation = A_crit_approximation * g;

##############################################################
# section: RETRIEVING GAUGE TRANSFORMATIONS AND R MATRICES
##############################################################

Atmp, lf, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars);
Atmp *= exp(lf);
Atmp, GH_tmp, GV_tmp, _ = fix_continuous_gauge(Atmp);
Atmp, _, DH_tmp, DV_tmp = fix_discrete_gauge(Atmp, accepted_elements; tol = 1e-7);

GH = GH_tmp;
GV = GV_tmp;
DH = DH_tmp;
DV = DV_tmp;
Rmatrices = py"Rmatrices";

##############################################################
# section: DEFINING N,S,W,E PIECES
##############################################################

py"""

def NSWE(Rs,direc,pars):
	spliteps = pars["gilt_eps"]*1e-3
		# TK change:
		# replaced the match statement by the 4 if statements below.
		# the match statement causes an error for TK, not for NE. WTF.
	if direc=="N":
		NRs=[]
		NLs=[]
		for R in Rs:
			NR, NL = R.split(0, 1, eps=spliteps, return_rel_err=False, return_sings=False)
			NRs.append(NR)
			NLs.append(NL)
		return ncon([NLs[0],NLs[1]],[[2,-2],[-1,2]]),ncon([NRs[1],NRs[0]],[[2,-2],[-1,2]])
	if direc=="S":
		SRs=[]
		SLs=[]
		for R in Rs:
			SL, SR = R.split(0, 1, eps=spliteps, return_rel_err=False, return_sings=False)
			SRs.append(SR)
			SLs.append(SL)
		return ncon([SLs[0],SLs[1]],[[-1,2],[2,-2]]), ncon([SRs[1],SRs[0]],[[-1,2],[2,-2]])
	if direc=="W":
		WTs=[]
		WBs=[]
		for R in Rs:
			WT, WB = R.split(0, 1, eps=spliteps, return_rel_err=False, return_sings=False)
			WTs.append(WT)
			WBs.append(WB)
		return ncon([WBs[0],WBs[1]],[[1,-2],[-1,1]]), ncon([WTs[1],WTs[0]],[[1,-2],[-1,1]])
	if direc=="E":
		ETs=[]
		EBs=[]
		for R in Rs:
			EB, ET = R.split(0, 1, eps=spliteps, return_rel_err=False, return_sings=False)
			ETs.append(ET)
			EBs.append(EB)
		return ncon([EBs[0],EBs[1]],[[-1,2],[2,-2]]), ncon([ETs[1],ETs[0]],[[-1,2],[2,-2]])
		# end TK change:
"""

Ns = [Rmatrices[("N", 1)], Rmatrices[("N", 2)]];
Ss = [Rmatrices[("S", 1)], Rmatrices[("S", 2)]];
Ws = [Rmatrices[("W", 1)], Rmatrices[("W", 2)]];
Es = [Rmatrices[("E", 1)], Rmatrices[("E", 2)]];

NL, NR = py"NSWE"(Ns, "N", gilt_pars);
SL, SR = py"NSWE"(Ss, "S", gilt_pars);
WB, WT = py"NSWE"(Ws, "W", gilt_pars);
EB, ET = py"NSWE"(Es, "E", gilt_pars);

##############################################################
# section: DEFINING A1,A2
##############################################################

tensors = Any[A_crit_approximation, SL, NR, WB, ET];
connects = Any[[1, 2, 3, 4], [3, -3], [1, -1], [-2, 2], [-4, 4]];
con_order = [4, 3, 2, 1];
A1 = ncon(tensors, connects, order = con_order);


tensors = Any[A_crit_approximation, NL, SR, EB, WT];
connects = Any[[1, 2, 3, 4], [-3, 3], [-1, 1], [2, -2], [4, -4]];
con_order = [4, 2, 3, 1];
A2 = ncon(tensors, connects, order = con_order);


##############################################################
# section: DEFINING t1_1,t1_2,t2_1,t2_2
##############################################################

py"""
def return_ts(A1,A2,pars):
	t1_1, t1_2 = A1.split([0,1], [2,3], chis=pars["cg_chis"], eps=pars["cg_eps"], return_rel_err=False,return_sings=False)
	t2_1, t2_2 = A2.split([2,1], [0,3], chis=pars["cg_chis"], eps=pars["cg_eps"], return_rel_err=False,return_sings=False) 
	return t1_1,t1_2,t2_1,t2_2
"""

t1_1, t1_2, t2_1, t2_2 = py"return_ts"(A1, A2, gilt_pars);

##############################################################
# section: DEFINING R,D,L,U
##############################################################

tttt = ncon([t1_1, t2_1, t1_2, t2_2], [[1, 4, -3], [1, 2, -4], [-1, 3, 2], [-2, 3, 4]]);
U_tmp, D_tmp, R_tmp, L_tmp = py"return_ts"(tttt, tttt, gilt_pars);

if rotate == false
	R = ncon([R_tmp, GH, DH.conj()], [[-1, -2, 3], [3, 4], [4, -3]])
	D = ncon([D_tmp, GV, DV.conj()], [[1, -2, -3], [1, 2], [2, -1]])
	L = ncon([L_tmp, GH.conj(), DH], [[1, -2, -3], [1, 2], [2, -1]])
	U = ncon([U_tmp, GV.conj(), DV], [[-1, -2, 3], [3, 4], [4, -3]])
else
	R = ncon([R_tmp, GV, DV.conj()], [[-1, -2, 3], [3, 4], [4, -3]])
	D = ncon([D_tmp, GH, DH.conj()], [[1, -2, -3], [1, 2], [2, -1]])
	L = ncon([L_tmp, GV.conj(), DV], [[1, -2, -3], [1, 2], [2, -1]])
	U = ncon([U_tmp, GH.conj(), DH], [[-1, -2, 3], [3, 4], [4, -3]])
end



##############################################################
# section: DEFINING h AND v - INDEX COMPRESSION MATRICES
##############################################################

tensors = Any[A_crit_approximation, A_crit_approximation, A_crit_approximation.conj(), A_crit_approximation.conj()];
connects = Any[[-2, 5, 3, 1], [-1, 1, 4, 6], [-4, 5, 3, 2], [-3, 2, 4, 6]];
con_order = [4, 6, 5, 3, 1, 2];
T1 = ncon(tensors, connects, order = con_order);
py"""
Stmp1, hR=($T1).eig([0,1],[2,3],chis=$(chi_compression),hermitian=True)
print(Stmp1.to_ndarray()[-1])
"""
hR = py"hR";

connects = Any[[3, 5, -2, 1], [4, 1, -1, 6], [3, 5, -4, 2], [4, 2, -3, 6]];
con_order = [3, 5, 4, 6, 1, 2];
T1 = ncon(tensors, connects, order = con_order);
py"""
Stmp2, hL=($T1).eig([0,1],[2,3],chis=$(chi_compression),hermitian=True)
print(Stmp2.to_ndarray()[-1])
"""
hL = py"hL";

connects = Any[[1, 4, 6, -4], [2, 4, 6, -2], [5, 3, 1, -3], [5, 3, 2, -1]];
con_order = [4, 6, 5, 3, 1, 2];
T2 = ncon(tensors, connects, order = con_order);
py"""
Stmp3, vT=($T2).eig([0,1],[2,3],chis=$(chi_compression),hermitian=True)
print(Stmp3.to_ndarray()[-1])
"""
vT = py"vT";

connects = Any[[1, -4, 6, 4], [2, -2, 6, 4], [5, -3, 1, 3], [5, -1, 2, 3]];
con_order = [6, 4, 5, 3, 1, 2];
T2 = ncon(tensors, connects, order = con_order);
py"""
Stmp4, vB=($T2).eig([0,1],[2,3],chis=$(chi_compression),hermitian=True)
print(Stmp4.to_ndarray()[-1])
"""
vB = py"vB";

#################################################################
# section: DEFINING THE SCALING OPERATOR
#################################################################

const global NRa = NR.to_ndarray();
const global SRa = SR.to_ndarray();
const global SLa = SL.to_ndarray();
const global NLa = NL.to_ndarray();
const global EBa = EB.to_ndarray();
const global WBa = WB.to_ndarray();
const global ETa = ET.to_ndarray();
const global WTa = WT.to_ndarray();
const global t1_1a = t1_1.to_ndarray();
const global t1_2a = t1_2.to_ndarray();
const global t2_2a = t2_2.to_ndarray();
const global t2_1a = t2_1.to_ndarray();
const global Da = D.to_ndarray();
const global Ua = U.to_ndarray();
const global Ra = R.to_ndarray();
const global La = L.to_ndarray();
const global hRa = hR.to_ndarray();
const global vTa = vT.to_ndarray();
const global hLa = hL.to_ndarray();
const global vBa = vB.to_ndarray();


NR = nothing;
SR = nothing;
SL = nothing;
NL = nothing;
EB = nothing;
WB = nothing;
ET = nothing;
WT = nothing;
t1_1 = nothing;
t1_2 = nothing;
t2_2 = nothing;
t2_1 = nothing;
D = nothing;
U = nothing;
R = nothing;
L = nothing;
hL = nothing;
vT = nothing;
hR = nothing;
vB = nothing;
tttt = nothing;


using TensorOperations

if rotate == false
	function scaling_operator(O)
		println("Contracting...")
		@time begin
			@tensor order = (9, 7, 1, 6, 2, 33, 12, 4, 8, 30, 28, 29, 42, 40, 36, 37, 25, 21, 39, 3, 11, 5, 43,
				15, 24, 27, 32, 10, 13, 22, 44, 35, 19, 16, 26, 14, 18, 31, 20, 41, 34, 17, 38, 23) begin
				res[-1, -2, -3, -4] :=
					NRa[32, 12] * SRa[10, 33] * SLa[36, 7] * NLa[9, 37] * EBa[39, 1] * WBa[2, 40] * ETa[4, 42] * WTa[43, 6] *
					t1_1a[9, 8, 21] * t1_1a[5, 6, 19] * t1_2a[13, 10, 11] * t1_2a[15, 3, 1] * t2_2a[24, 3, 2] * t2_2a[22, 7, 8] *
					t2_1a[12, 11, 16] * t2_1a[5, 4, 18] * Da[27, 15, 14] * Da[28, 23, 24] * Ua[17, 18, 30] * Ua[19, 20, 31] *
					Ra[13, 14, 26] * Ra[17, 16, 25] * La[29, 23, 22] * La[44, 21, 20] * O[34, 38, 35, 41] * hRa[32, 33, 34] *
					hLa[37, 36, 35] * vTa[39, 40, 38] * vBa[42, 43, 41] * hRa[25, 26, -1] * hLa[44, 29, -3] * vTa[27, 28, -2] *
					vBa[30, 31, -4]
			end
		end
		println("Contracting is over")
		return res
	end
else
	function scaling_operator(O)
		println("Contracting...")
		@time begin
			@tensor order = (36, 1, 25, 9, 33, 26, 4, 7, 40, 38, 44, 41, 30, 18, 6, 11, 2, 32, 3, 15, 29, 8, 22,
				12, 10, 24, 37, 21, 43, 16, 13, 39, 14, 28, 35, 5, 42, 19, 20, 34, 27, 17, 31, 23) begin
				res[-1, -2, -3, -4] :=
					NRa[25, 12] * SRa[10, 26] * SLa[29, 7] * NLa[9, 30] * EBa[32, 1] * WBa[2, 33] * ETa[4, 35] * WTa[36, 6] * t1_1a[9, 8, 21] * t1_1a[5, 6, 19] * t1_2a[13, 10, 11] * t1_2a[15, 3, 1] * t2_2a[24, 3, 2] * t2_2a[22, 7, 8] * t2_1a[12, 11, 16] *
					t2_1a[5, 4, 18] * Da[37, 15, 14] * Da[38, 23, 24] * Ua[17, 18, 42] * Ua[19, 20, 41] * Ra[13, 14, 40] * Ra[17, 16, 39] * La[43, 23, 22] * La[44, 21, 20] * O[27, 31, 28, 34] * hLa[25, 26, 27] * hRa[30, 29, 28] * vTa[32, 33, 31] *
					vBa[35, 36, 34] *
					hLa[37, 38, -1] * hRa[42, 41, -3] * vTa[43, 44, -2] * vBa[40, 39, -4]

			end
		end
		println("Contracting is over")
		return res
	end
end



##############################################################
# section: COMPUTING THE EIGENVALUES
##############################################################

O = 2 .* rand(chi_compression, chi_compression, chi_compression, chi_compression) .- 1;

res = eigsolve(scaling_operator, O, N, :LM; verbosity = verbosity, issymmetric = false, ishermitian = false, krylovdim = krylovdim);

println("SCALING DIMENSIONS FROM THE SCALING OPERATOR FOR chi_compression=$chi_compression:")

if path_to_tensor != "none"
	println("path to tensor: ", path_to_tensor)
end

for el in res[1]
	println(round(-log(2, abs(el)), digits = 4), "  ", round(real(el), digits = 3))
end
println("END OF LIST FOR chi_compression=$chi_compression:")

if path_to_tensor != "none"
	result = Dict("A" => A_crit_approximation,
		"recursion_depth" => gilt_pars["recursion_depth"],
		"eigensystem" => res,
	)
else
	result = Dict("A" => A_crit_approximation,
		"eigensystem" => res,
	)
end

filename = "DSO/" * gilt_pars_identifier(gilt_pars) * "_rotate=($rotate)"

if path_to_tensor != "none"
	filename *= ("_path=" * path_to_tensor)
else
	if abs(Jratio - 1) > 1.e-10
		filename = filename * "_Jratio=$(Jratio)"
	end
end

filename *= ".data"

serialize(filename, result)

