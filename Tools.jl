# topics:
# - GENERIC PACKAGES
# - PYTHON CODES
# - BASIC HANDLING OF TRAJECTORIES
# - TRAJECTORIES PLOTTING
# - NUMERICAL DIFFERENTIATION
# - CRITICAL TEMPERATURE SEARCH
# - ROUTINES FOR ANALYSIS
# - PLOTTING SHORTHANDS
# - EXACT ISING SPECTRUM


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



########################################################
# topic: BASIC HANDLING OF TRAJECTORIES
########################################################

function initial_tensor(relT)
    py"get_initial_tensor"(Dict("beta" => log(1 + sqrt(2)) / 2 / relT, "symmetry_tensors" => true))
end

function gilt_pars_identifier(pars)
    return "rotate=$(pars["rotate"])_$(pars["cg_chis"][end])_$(pars["gilt_eps"])_$(pars["cg_eps"])"
end

function form_the_file_name(relT, len::Int64, gilt_pars)
    return gilt_pars_identifier(gilt_pars) * "__relT=$(relT)_len=$len"
end

function form_the_no_len_dot_data_pattern(relT, gilt_pars)
    gilt_eps = gilt_pars["gilt_eps"]
    chi = gilt_pars["cg_chis"][end]
    cg_eps = gilt_pars["cg_eps"]
    rotate = gilt_pars["rotate"]
    Regex("rotate=\\Q$rotate\\E_\\Q$chi\\E_\\Q$gilt_eps\\E_\\Q$cg_eps\\E__\\QrelT=$relT\\E_len=(.*).data")
end

function handle_the_database(relT::t, len::Int64, gilt_pars) where {t}
    trajectories = readdir("trajectories/")
    pattern = form_the_no_len_dot_data_pattern(relT, gilt_pars)
    matching = filter(x -> occursin(pattern, x), trajectories)

    verbose = gilt_pars["verbosity"] > 1

    if length(matching) == 0
        if verbose
            @info "No record in the database, will compute the trajectory from scratch"
        end
        return Any[initial_tensor(relT)], t[0.0], Array{t,1}[[0, 0, 0, 0, 0]], 0
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
        file_name = form_the_file_name(relT, existing_length, gilt_pars)
        traj = deserialize("trajectories/" * file_name * ".data")
        len = min(len, existing_length)
        return traj["A"][1:(len+1)], traj["log_fact"][1:(len+1)], traj["errs"][1:(len+1)], existing_length
    elseif length(mathcing) > 2
        @warn "The database is corrupted (repeated entries), will recompute the result from scratch."
        return Any[initial_tensor(relT)], t[0.0], Array{t,1}[[0, 0, 0, 0, 0]], 0
    end
end

function trajectory(relT::t, len::Int64, gilt_pars) where {t}
    A_hist, log_fact_hist, errs_hist, existing_length = handle_the_database(relT, len, gilt_pars)
    if existing_length >= len
        return Dict(
            "A" => A_hist,
            "log_fact" => log_fact_hist,
            "errs" => errs_hist
        )
    end
    file_name = form_the_file_name(relT, len, gilt_pars)
    if existing_length != 0
        file_name_old = form_the_file_name(relT, existing_length, gilt_pars)
        Filesystem.mv("trajectories/" * file_name_old * ".log", "trajectories/" * file_name * ".log")
    end

    out_log = open("trajectories/" * file_name * ".log", "a")

    redirect_stdio(stdout=out_log) do
        for i = (existing_length+1):len
            A, log_fact, errs = py"gilttnr_step"(A_hist[i], log_fact_hist[i], gilt_pars)
            push!(A_hist, A)
            push!(log_fact_hist, log_fact)
            push!(errs_hist, errs)
        end
    end

    traj = Dict(
        "A" => A_hist,
        "log_fact" => log_fact_hist,
        "errs" => errs_hist
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
    for i = 2:len
        A, log_fact, errs = py"gilttnr_step"(A_hist[i-1], log_fact_hist[i-1], gilt_pars)
        push!(A_hist, A)
        push!(log_fact_hist, log_fact)
        push!(errs_hist, errs)
    end


    traj = Dict(
        "A" => A_hist,
        "log_fact" => log_fact_hist,
        "errs" => errs_hist
    )

    return traj
end



function trajectory_no_saving(relT::t, len::Int64, gilt_pars) where {t}
    A_hist, log_fact_hist, errs_hist, existing_length = handle_the_database(relT, len, gilt_pars)
    if existing_length >= len
        return Dict(
            "A" => A_hist,
            "log_fact" => log_fact_hist,
            "errs" => errs_hist
        )
    end

    for i = (existing_length+1):len
        A, log_fact, errs = py"gilttnr_step"(A_hist[i], log_fact_hist[i], gilt_pars)
        push!(A_hist, A)
        push!(log_fact_hist, log_fact)
        push!(errs_hist, errs)
    end

    traj = Dict(
        "A" => A_hist,
        "log_fact" => log_fact_hist,
        "errs" => errs_hist
    )

    return traj
end


########################################################
# topic: TRAJECTORIES PLOTTING
########################################################

using CairoMakie

function N_first_elements_zero_if_missing(array::Vector{t}, N=2) where {t}
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

function plot_the_trajectory(relT::Float64, gilt_pars_local; traj_len, N=20)

    traj = trajectory(relT, traj_len, gilt_pars_local)

    spectrum = traj["A"] .|> py"get_A_spectrum" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix


    fig = Figure(; size=(400, 400))

    ax = Axis(
        fig[1, 1],
        title="trajectory of singular values",
        ylabel="singular values",
        xlabel="step")

    for eigenvalue_index = 1:N
        lines!(ax, 0:traj_len, spectrum[eigenvalue_index, :], color=:black)
        scatter!(ax, 0:traj_len, spectrum[eigenvalue_index, :], color=:red, markersize=6)
    end

    file_name = form_the_file_name(relT, traj_len, gilt_pars_local)
    save("trajectory_plots/" * file_name * "_traj_plot_of_$(N)_singular_values.pdf", fig)
    return fig
end

function plot_the_trajectory(A_init::PyObject, gilt_pars_local; traj_len, N=20)

    traj = trajectory(A_init, traj_len, gilt_pars_local)

    spectrum = traj["A"] .|> py"get_A_spectrum" .|> (x -> N_first_elements_zero_if_missing(x, N)) |> vec_of_vec_to_matrix


    fig = Figure(; size=(500, 500))

    ax = Axis(
        fig[1, 1],
        title="trajectory of singular values",
        ylabel="singular values",
        xlabel="step")

    for eigenvalue_index = 1:N
        lines!(ax, 1:traj_len, spectrum[eigenvalue_index, :], color=:black)
        scatter!(ax, 1:traj_len, spectrum[eigenvalue_index, :], color=:red, markersize=6)
    end

    fig
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

function phase(A, gilt_pars; max_number_of_steps=40, tol=1e-4)
    for i = 1:max_number_of_steps
        A, _ = py"gilttnr_step"(A, 1.0, gilt_pars)
        second_eigenvalue = (A|>py"get_A_spectrum"|>N_first_elements_zero_if_missing)[2]
        if abs(second_eigenvalue - 1) < tol
            return LOW_TEMPERATURE, i
        elseif abs(second_eigenvalue) < tol
            return HIGH_TEMPERATURE, i
        end
    end
    return UNDETERMINED, max_number_of_steps
end

function perform_search(relT_low, relT_high, gilt_pars; search_tol=1e-5, max_number_of_steps=40, phase_tol=1e-4, verbose=false)
    test_l = phase(initial_tensor(relT_low), gilt_pars, max_number_of_steps=max_number_of_steps, tol=phase_tol)[1] == LOW_TEMPERATURE
    test_h = phase(initial_tensor(relT_high), gilt_pars, max_number_of_steps=max_number_of_steps, tol=phase_tol)[1] == HIGH_TEMPERATURE
    if !test_l || !test_h
        @warn "No signs of criticality, try another range"
        return relT_low, relT_high, 0
    end
    if verbose
        @info "Sanity check passed"
    end
    relT = (relT_low + relT_high) / 2
    gap = relT_high - relT_low
    len = 0
    while gap > search_tol
        if verbose
            @info "Checking relT=$relT"
        end
        phase_res, len = phase(initial_tensor(relT), gilt_pars, max_number_of_steps=max_number_of_steps, tol=phase_tol)
        if phase_res == LOW_TEMPERATURE
            relT_low = relT
            verbose ? (@info "It is LOW_TEMPERATURE") : nothing
        elseif phase_res == HIGH_TEMPERATURE
            relT_high = relT
            verbose ? (@info "It is HIGH_TEMPERATURE") : nothing
        elseif phase_res == UNDETERMINED
            @warn "Cannot resolve the phase of the tensor anymore, try using longer trajectories"
            return relT_low, relT_high, len
        end
        relT = (relT_low + relT_high) / 2
        gap = relT_high - relT_low
    end
    return relT_low, relT_high, len
end


########################################################
# topic: ROUTINES FOR ANALYSIS
########################################################

function variation(v::Vector)
    v[1:(end-1)] - v[2:end]
end

function relative_variation(v::Vector)
    (v[1:(end-1)] - v[2:end]) ./ norm.(v[1:end])
end


########################################################
# topic: PLOTTING SHORTHANDS
########################################################


function yloglines(data...)
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10)
    lines!(ax, data...)
    return fig
end

function xloglines(data...)
    fig = Figure()
    ax = Axis(fig[1, 1], xscale=log10)
    lines!(ax, data...)
    return fig
end

function xyloglines(data...)
    fig = Figure()
    ax = Axis(fig[1, 1], xscale=log10, yscale=log10)
    lines!(ax, data...)
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
    for i = eachindex(exact_degeneracies)
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
    for i = eachindex(exact_degeneracies_even)
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
