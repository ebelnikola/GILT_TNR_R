# initial_tensor

# topics:
# - GENERIC PACKAGES
# - PYTHON CODES
# - MISCELLANEOUS: various plot labels formatters, N_first_elements_zero_if_missing, vec_of_vec_to_matrix, name_of_index_in_Z2_notation, variation, relative_variation
# - BASIC HANDLING OF TRAJECTORIES: initial_tensor, trajectory (2 methods)
# - TRAJECTORIES PLOTTING: plot_the_trajectory (2 methods)
# - NUMERICAL DIFFERENTIATION: df - performs differentiation
# - CRITICAL TEMPERATURE SEARCH: perform_search
# - PLOTTING SHORTHANDS: yloglines, xloglines, xyloglines
# - EXACT ISING SPECTRUM: exact_spectrum, exact_spectrum_even


mkpath("critical_temperatures")
mkpath("diff_tests")
mkpath("eigensystems")
mkpath("newton")
mkpath("trajectories")
mkpath("trajectory_plots")
mkpath("DSO");


########################################################
# topic: GENERIC PACKAGES
########################################################

using Base.Filesystem, LinearAlgebra, Serialization, ArgParse

########################################################
# topic: PYTHON CODES
########################################################

using PyCall
pushfirst!(pyimport("sys")."path", "GiltTNR")
@pyinclude("GiltTNR/GiltTNR2D_essentials.py")


###########################
# topic: MISCELLANEOUS
###########################

using LaTeXStrings

function float_scientific_LaTeX(values; sigdigits = 1)
	[L"%$(round(value;sigdigits=sigdigits))" for value in values]
end

function int_LaTeX(values)
	[L"%$(Int(value))" for value in values]
end

function int_powers_of_10_LaTeX(values)
	[L"10^{%$(Int(log10(value)))}" for value in values]
end

function int_powers_of_2_LaTeX(values)
	[L"2^{%$(Int(log2(value)))}" for value in values]
end

function int_power_of_10_scientific_LaTeX(value)
	if value < 1e-4
		return L"1e{%$(Int(log10(value)))}"
	elseif value > 1e4
		return L"1e\,{%$(Int(log10(value)))}"
	elseif value >= 1.0
		return L"%$(Int(value))"
	else
		return L"{%$value}"
	end
end

function int_powers_of_10_scientific_LaTeX(values)
	[int_power_of_10_scientific_LaTeX(value) for value in values]
end


function N_first_elements_zero_if_missing(array::Vector{t}, N = 2) where {t}
	len = length(array)
	if len < N
		return vcat(array, zeros(t, N - len))
	else
		return array[1:N]
	end
end

function vec_of_vec_to_matrix(vec)
	mat = zeros(eltype(vec[1]), length(vec[1]), length(vec))
	for i in eachindex(vec)
		mat[:, i] .= vec[i]
	end
	return mat
end


function bit_to_sect(a)
	if a == 0
		return "+"
	else
		return "-"
	end
end

function name_of_index_in_Z2_notation(index, chi = 30)
	chi = chi ÷ 2
	section = ((index[1] - 1) ÷ chi, (index[2] - 1) ÷ chi, (index[3] - 1) ÷ chi, (index[4] - 1) ÷ chi) .|> bit_to_sect

	position = ((index[1] - 1) % chi + 1, (index[2] - 1) % chi + 1, (index[3] - 1) % chi + 1, (index[4] - 1) % chi + 1)
	string(position[1]) * string(section[1]) * "," * string(position[2]) * string(section[2]) * "," * string(position[3]) * string(section[3]) * "," * string(position[4]) * string(section[4])
end


function variation(v::Vector)
	v[1:(end-1)] - v[2:end]
end

function relative_variation(v::Vector)
	(v[1:(end-1)] - v[2:end]) ./ norm.(v[1:end])
end



########################################################
# topic: BASIC HANDLING OF TRAJECTORIES
########################################################

# TK change
function initial_tensor(initialA_pars)
	#   Ratio of couplings is Jratio and relT=1 is the critical point. 

	relT = initialA_pars["relT"]
	Jratio = initialA_pars["Jratio"]
	A_0 = py"get_initial_tensor_aniso"(Dict("relT" => relT, "Jratio" => Jratio, "symmetry_tensors" => true))
	return A_0
end

function gilt_pars_identifier(pars)
	return "rotate=$(pars["rotate"])_$(pars["cg_chis"][end])_$(pars["gilt_eps"])_$(pars["cg_eps"])"
end

function initialA_pars_identifier(pars)
	# if Jratio == 1 we include only relT for compatibility with earlier data 
	if abs(pars["Jratio"] - 1.0) < 1.e-10
		return "__relT=$(pars["relT"])"
	else
		return "__relT=$(pars["relT"])_Jratio=$(pars["Jratio"])"
	end
end

function form_the_file_name(initialA_pars, len::Int64, gilt_pars)
	return gilt_pars_identifier(gilt_pars) * initialA_pars_identifier(initialA_pars) * "_len=$len"
end

function form_the_no_len_dot_data_pattern(initialA_pars, gilt_pars)
	gilt_eps = gilt_pars["gilt_eps"]
	chi = gilt_pars["cg_chis"][end]
	cg_eps = gilt_pars["cg_eps"]
	rotate = gilt_pars["rotate"]
	relT = initialA_pars["relT"]
	Jratio = initialA_pars["Jratio"]

	# if Jratio==1, Jratio is not included in the filename
	if abs(Jratio - 1.0) < 1.e-10
		pattern = Regex("rotate=\\Q$rotate\\E_\\Q$chi\\E_\\Q$gilt_eps\\E_\\Q$cg_eps\\E__\\QrelT=$relT\\E_len=(.*).data")
	else
		pattern = Regex("rotate=\\Q$rotate\\E_\\Q$chi\\E_\\Q$gilt_eps\\E_\\Q$cg_eps\\E__\\QrelT=$relT\\E_\\QJratio=$Jratio\\E_len=(.*).data")
	end

	return pattern
end


function handle_the_database(initialA_pars, len::Int64, gilt_pars)
	trajectories = readdir("trajectories/")
	pattern = form_the_no_len_dot_data_pattern(initialA_pars, gilt_pars)
	matching = filter(x -> occursin(pattern, x), trajectories)

	t = Float64

	verbose = gilt_pars["verbosity"] > 1

	if length(matching) == 0
		if verbose
			@info "No record in the database, will compute the trajectory from scratch"
		end
		return Any[initial_tensor(initialA_pars)], t[0.0], Array{t, 1}[[0, 0, 0, 0, 0]], 0
	elseif length(matching) == 1
		existing_length = parse(Int64, match(pattern, matching[1]).captures[1])
		if existing_length >= len
			if verbose
				@info "Found the record with len=$existing_length"
			end
		else
			if verbose
				@info "Found the record with len=$existing_length, extending..."
			end
		end
		file_name = form_the_file_name(initialA_pars, existing_length, gilt_pars)
		traj = deserialize("trajectories/" * file_name * ".data")
		len = min(len, existing_length)
		return traj["A"][1:(len+1)], traj["log_fact"][1:(len+1)], traj["errs"][1:(len+1)], existing_length
	elseif length(matching) > 2
		@warn "The database is corrupted (repeated entries), will recompute the result from scratch."
		return Any[initial_tensor(initialA_pars)], t[0.0], Array{t, 1}[[0, 0, 0, 0, 0]], 0
	end
end

function trajectory(initialA_pars, len::Int64, gilt_pars)
	A_hist, log_fact_hist, errs_hist, existing_length = handle_the_database(initialA_pars, len, gilt_pars)
	if existing_length >= len
		return Dict(
			"A" => A_hist,
			"log_fact" => log_fact_hist,
			"errs" => errs_hist,
		)
	end
	file_name = form_the_file_name(initialA_pars, len, gilt_pars)
	if existing_length != 0
		file_name_old = form_the_file_name(initialA_pars, existing_length, gilt_pars)
		Filesystem.mv("trajectories/" * file_name_old * ".log", "trajectories/" * file_name * ".log")
	end

	out_log = open("trajectories/" * file_name * ".log", "a")

	redirect_stdio(stdout = out_log) do
		for i ∈ (existing_length+1):len
			A, log_fact, errs = py"gilttnr_step"(A_hist[i], log_fact_hist[i], gilt_pars)
			push!(A_hist, A)
			push!(log_fact_hist, log_fact)
			push!(errs_hist, errs)
		end
	end

	traj = Dict(
		"A" => A_hist,
		"log_fact" => log_fact_hist,
		"errs" => errs_hist,
	)
	serialize("trajectories/" * file_name * ".data", traj)
	if existing_length != 0
		Filesystem.rm("trajectories/" * file_name_old * ".data")
	end
	return traj
end

function trajectory(A_init::PyObject, len::Int64, gilt_pars)
	A_hist = [A_init]
	log_fact_hist = Float64[0.0]
	errs_hist = Vector{Float64}[[0, 0, 0, 0, 0]]
	for i ∈ 2:len
		A, log_fact, errs = py"gilttnr_step"(A_hist[i-1], log_fact_hist[i-1], gilt_pars)
		push!(A_hist, A)
		push!(log_fact_hist, log_fact)
		push!(errs_hist, errs)
	end


	traj = Dict(
		"A" => A_hist,
		"log_fact" => log_fact_hist,
		"errs" => errs_hist,
	)

	return traj
end


########################################################
# topic: TRAJECTORIES PLOTTING
########################################################

using CairoMakie

function plot_the_trajectory(initialA_pars_local, gilt_pars_local; traj_len, N = 20)

	traj = trajectory(initialA_pars_local, traj_len, gilt_pars_local)

	spectrum = traj["A"] .|> py"get_A_spectrum" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix


	# TK addition
	if false #NE - supressed this printing 
		for eig ∈ 1:N
			for irg ∈ 1:traj_len
				println(irg, "  ", spectrum[eig, irg])
			end
			println("  ")
		end
	end

	fig = Figure(; size = (400, 400))

	ax = Axis(
		fig[1, 1],
		ytickformat = float_scientific_LaTeX,
		xtickformat = int_LaTeX,
		ylabel = L"\text{singular value}",
		xlabel = L"\text{RG step}")

	for eigenvalue_index ∈ 1:N
		lines!(ax, 0:traj_len, spectrum[eigenvalue_index, :], color = :black)
		scatter!(ax, 0:traj_len, spectrum[eigenvalue_index, :], color = :red, markersize = 6)
	end

	file_name = form_the_file_name(initialA_pars_local, traj_len, gilt_pars_local)
	save("trajectory_plots/" * file_name * "_traj_plot_of_$(N)_singular_values.pdf", fig)
	return fig
end



function plot_the_trajectory(traj; N = 20)

	traj_len = length(traj)
	spectrum = traj .|> py"get_A_spectrum" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix


	fig = Figure(; size = (400, 400))

	ax = Axis(
		fig[1, 1],
		ytickformat = float_scientific_LaTeX,
		xtickformat = int_LaTeX,
		ylabel = L"\text{singular value}",
		xlabel = L"\text{RG step}")

	for eigenvalue_index ∈ 1:N
		lines!(ax, 0:traj_len-1, spectrum[eigenvalue_index, :], color = :black)
		scatter!(ax, 0:traj_len-1, spectrum[eigenvalue_index, :], color = :red, markersize = 6)
	end

	return fig, ax
end


function plot_the_trajectory(A_init::PyObject, gilt_pars_local; traj_len, N = 20)

	traj = trajectory(A_init, traj_len, gilt_pars_local)

	spectrum = traj["A"] .|> py"get_A_spectrum" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix


	fig = Figure(; size = (500, 500))

	ax = Axis(
		fig[1, 1],
		ytickformat = float_scientific_LaTeX,
		xtickformat = int_LaTeX,
		ylabel = L"\text{singular value}",
		xlabel = L"\text{step}")

	for eigenvalue_index ∈ 1:N
		lines!(ax, 1:traj_len, spectrum[eigenvalue_index, :], color = :black)
		scatter!(ax, 1:traj_len, spectrum[eigenvalue_index, :], color = :red, markersize = 6)
	end

	fig
end


function find_trajectory_eigenvalues(initialA_pars_local, gilt_pars_local; RG_step, N = 20)
	# Extract the N eigenvalues we use to visualize the trajectory at RG step RG_step
	traj = trajectory(initialA_pars_local, RG_step, gilt_pars_local)

	spectrum = traj["A"] .|> py"get_A_spectrum" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix
	single_spectrum = spectrum[:, RG_step]

	spectrum_flipped = traj["A"] .|> py"get_A_spectrum_flipped" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix
	single_spectrum_flipped = spectrum_flipped[:, RG_step]

	return single_spectrum, single_spectrum_flipped
end

########################################################
# topic: NUMERICAL DIFFERENTIATION
########################################################

if !(@isdefined(NumDifferentiation))
	include("NumDifferentiation.jl")
end

using Main.NumDifferentiation


########################################################
# topic: CRITICAL TEMPERATURE SEARCH
########################################################

@enum Phase begin
	HIGH_TEMPERATURE
	LOW_TEMPERATURE
	UNDETERMINED
end

function phase(A, gilt_pars; max_number_of_steps = 40, tol = 1e-4)
	for i ∈ 1:max_number_of_steps
		A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)
		second_eigenvalue = (A|>py"get_A_spectrum"|>N_first_elements_zero_if_missing)[2]
		if abs(second_eigenvalue - 1) < tol
			return LOW_TEMPERATURE, i
		elseif abs(second_eigenvalue) < tol
			return HIGH_TEMPERATURE, i
		end
	end
	return UNDETERMINED, max_number_of_steps
end

function perform_search(initialA_pars_low, initialA_pars_high, gilt_pars; search_tol = 1e-5, max_number_of_steps = 40, phase_tol = 1e-4, verbose = false)
	test_l = phase(initial_tensor(initialA_pars_low), gilt_pars, max_number_of_steps = max_number_of_steps, tol = phase_tol)[1] == LOW_TEMPERATURE
	test_h = phase(initial_tensor(initialA_pars_high), gilt_pars, max_number_of_steps = max_number_of_steps, tol = phase_tol)[1] == HIGH_TEMPERATURE
	if !test_l || !test_h
		@warn "No signs of criticality, try another range"
		return initialA_pars_low, initialA_pars_high, 0
	end
	if verbose
		@info "Sanity check passed"
	end
	# it is assumed that Jratio is constant
	Jratio = initialA_pars_high["Jratio"]

	relT_low = initialA_pars_low["relT"]
	relT_high = initialA_pars_high["relT"]
	relT = (relT_low + relT_high) / 2
	gap = relT_high - relT_low
	len = 0
	while gap > search_tol
		relT_low = initialA_pars_low["relT"]
		relT_high = initialA_pars_high["relT"]
		relT = (relT_low + relT_high) / 2
		initialA_pars = Dict("relT" => relT, "Jratio" => Jratio)

		if verbose
			@info "Compute phase for relT=$relT"
		end

		phase_res, len = phase(initial_tensor(initialA_pars), gilt_pars, max_number_of_steps = max_number_of_steps, tol = phase_tol)
		if phase_res == LOW_TEMPERATURE
			# initialA_pars_low["relT"] = initialA_pars["relT"]
			initialA_pars_low = copy(initialA_pars)
			verbose ? (@info "It is LOW_TEMPERATURE") : nothing
		elseif phase_res == HIGH_TEMPERATURE
			# initialA_pars_high["relT"] = initialA_pars["relT"]
			initialA_pars_high = copy(initialA_pars)
			verbose ? (@info "It is HIGH_TEMPERATURE") : nothing
		elseif phase_res == UNDETERMINED
			@warn "Cannot resolve the phase of the tensor anymore, try using longer trajectories"
			return initialA_pars_low, initialA_pars_high, len
		end
		gap = initialA_pars_high["relT"] - initialA_pars_low["relT"]

		if verbose
			@info "gap=$gap"
		end

	end
	return initialA_pars_low, initialA_pars_high, len
end


function find_critical_temperature(chi, gilt_pars, search_tol, Jratio)
	# Extract critical relT from a previous search in critical_temperatures/
	# We return three relT's - first two give an interval, third is the midpoint
	# If case is not found we return 0.,0.,0.

	verbose = gilt_pars["verbosity"] > 1

	# We only include Jratio in the file name if Jratio is not 1
	# This insures backwards compatibility with earlier file names
	filename = "critical_temperatures/" * gilt_pars_identifier(gilt_pars) * "__tol=$(search_tol)"
	if abs(Jratio - 1) > 1.e-10
		filename = filename * "_Jratio=$(Jratio)"
	end
	filename = filename * ".data"

	if verbose
		@info "find_critical_temperature: searched for " * filename
	end

	if isfile(filename)
		crit_pars = deserialize(filename)
		relT_low = crit_pars[1]["relT"]
		relT_high = crit_pars[2]["relT"]
		relT_mid = (relT_low + relT_high) / 2
		if verbose
			@info "found  critical_temperature" * string(relT_mid)
		end
		return relT_low, relT_high, relT_mid
	else
		return 0.0, 0.0, 0.0
	end
end


########################################################
# topic: PLOTTING SHORTHANDS
########################################################


function yloglines(data...)
	fig = Figure()
	ax = Axis(fig[1, 1], yscale = log10)
	lines!(ax, data...)
	return fig
end

function xloglines(data...)
	fig = Figure()
	ax = Axis(fig[1, 1], xscale = log10)
	lines!(ax, data...)
	return fig
end

function xyloglines(data...)
	fig = Figure()
	ax = Axis(fig[1, 1], xscale = log10, yscale = log10)
	lines!(ax, data...)
	return fig
end

function ylogscatter(data...)
	fig = Figure()
	ax = Axis(fig[1, 1], yscale = log10)
	scatter!(ax, data...)
	return fig
end

function xlogscatter(data...)
	fig = Figure()
	ax = Axis(fig[1, 1], xscale = log10)
	scatter!(ax, data...)
	return fig
end

function xylogscatter(data...)
	fig = Figure()
	ax = Axis(fig[1, 1], xscale = log10, yscale = log10)
	scatter!(ax, data...)
	return fig
end

import CairoMakie: lines, scatter
function lines(X::Vector, Y::Matrix, args...)
	fig = Figure()
	ax = Axis(fig[1, 1])
	for i in axes(Y, 2)
		lines!(ax, X, Y[:, i])
	end
	return fig
end

function scatter(X::Vector, Y::Matrix, args...)
	fig = Figure()
	ax = Axis(fig[1, 1])
	for i in axes(Y, 2)
		scatter!(ax, X, Y[:, i])
	end
	return fig
end


########################################################
# topic: EXACT ISING SPECTRUM
########################################################

spec = deserialize("IsingExactLevels")

if !@isdefined(exact_degeneracies)
	const global exact_degeneracies = Int64.(spec[1, :])
	const global exact_levels = spec[2, :]
	const global exact_spectrum_max_number_of_levels = sum(exact_degeneracies)
end

function exact_spectrum(n)
	spec = Float64[]
	flag = false
	for i ∈ eachindex(exact_degeneracies)
		levels_to_append = exact_degeneracies[i]
		if length(spec) + levels_to_append > n
			levels_to_append = n - length(spec)
			flag = true
		end
		append!(spec, fill(exact_levels[i], levels_to_append))
		if flag
			break
		end
	end
	return spec
end

spec = deserialize("IsingEvenExactLevels")

if !@isdefined(exact_degeneracies_even)
	const global exact_degeneracies_even = Int64.(spec[1, :])
	const global exact_levels_even = spec[2, :]
	const global exact_spectrum_max_number_of_levels_even = sum(exact_degeneracies_even)
end

function exact_spectrum_even(n)
	spec = Float64[]
	flag = false
	for i ∈ eachindex(exact_degeneracies_even)
		levels_to_append = exact_degeneracies_even[i]
		if length(spec) + levels_to_append > n
			levels_to_append = n - length(spec)
			flag = true
		end
		append!(spec, fill(exact_levels_even[i], levels_to_append))
		if flag
			break
		end
	end
	return spec
end


########################################################
# topic: RG SPECTRUM - add TK 5/20
########################################################

function find_RG_spectrum(gilt_pars, Z2_odd_sector, freeze_R, Jratio; short = false)
	# Searches previously computed data in eigensystems/ for case matching
	# the parameters: gilt_pars,Z2_odd_sector,freeze_R,Jratio
	# Returns array of eigenvalues if found.
	# If case is not found we return [0.]
	# If short==true we search in eigensystems/short/

	filename = "eigensystems/"
	if short == true
		filename = filename * "short/"
	end

	filename = filename * gilt_pars_identifier(gilt_pars) * "__Z2_odd_sector=$(Z2_odd_sector)_freeze_R=$(freeze_R)"
	if abs(Jratio - 1.0) > 1.e-10
		filename = filename * "_Jratio=$(Jratio)"
	end
	filename = filename * "_eigensystem.data"

	if isfile(filename)
		result = deserialize(filename)
		vals, vecs, info = result["eigensystem"]
	else
		vals = [0.0]
	end

	return vals
end


