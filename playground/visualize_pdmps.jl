#=

    TODO:

    More targets:
        - [ ] 2D Gaussian
        - [ ] 2D Cauchy

    Improve Boomerang:
        - [ ] better covariance matrix
        - [ ] adaptive refreshment rate?

    Visualization:
        - [ ] plot velocity as arrows? Only interesting for Boomerang and BouncyParticle
        - [ ] highlight refreshment events with a different marker type
        - [ ] slightly fade the past trajectory so that the current position always stands out.
        - [ ] marginal histograms are somewhat ugly at the beginning.

    Other:
        - [ ] Quick and dirty Turing integration? Would be nice to also show NUTS/ RWMH/ HMC results for comparison.

=#

using PDMPSamplers
using GLMakie
using Makie
import DifferentiationInterface as DI
import ForwardDiff
include("mcmc_demo.jl")
import Random
import Statistics
using Test
include("../test/helper-gen_data.jl")

using Turing#, MCMCDiagnosticTools
import DynamicPPL, AbstractMCMC, Bijectors, ForwardDiff, LazyArrays
import FillArrays
root = "/home/don/hdd/surfdrive/Postdoc/TutorialRafteryGottardo/julia_src/"
include(joinpath(root, "moms_functions.jl"))

# include("/home/don/hdd/surfdrive/Postdoc/Talks/2025_London_CFE_Statistics/lr_functions.jl")

function fit_target(d, pdmp_type; T =  20_000, show_progress = true)

    testval_x = [1.124, -.876]
    testval_v = [0.1, 0.2]

    @assert DI.gradient(x -> logpdf(d, x), DI.AutoForwardDiff(), testval_x) ≈ gradlogpdf(d, testval_x)
    @assert DI.hessian( x -> logpdf(d, x), DI.AutoForwardDiff(), testval_x) * testval_v ≈ hessian_vector_product(d, testval_x, testval_v)

    ∇f!(out, x) = out .= .-gradlogpdf(d, x)
    ∇²f!(out, x, v) = out .= .-hessian_vector_product(d, x, v)

    flow = pdmp_type(I(2), zeros(2))

    alg = GridThinningStrategy(; hvp = ∇²f!)
    x0 = randn(2)
    θ0 = PDMPSamplers.initialize_velocity(flow, length(x0))
    ξ0 = SkeletonPoint(x0, θ0)

    grad = FullGradient(∇f!)
    trace0, stats0 = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)

    time_and_positions = collect(trace0)
    times = first.(time_and_positions)
    positions = getindex.(time_and_positions, 2)

    dtrace = PDMPDiscretize(trace0, 0.1)
    dtime_and_positions = collect(dtrace)
    dtimes = first.(dtime_and_positions)
    dpositions = getindex.(dtime_and_positions, 2)

    # remove duplicates
    dtimes_not_in_times = findall(t -> t ∉ times, dtimes)
    dtimes = dtimes[dtimes_not_in_times]
    dpositions = dpositions[dtimes_not_in_times]

    mtimes = [times; dtimes]
    o = sortperm(mtimes)
    mpositions = [positions; dpositions][o]
    mtimes = mtimes[o]

    itest  = 5
    ifound = findfirst(==(times[itest]), mtimes)
    @assert times[itest] == mtimes[ifound] && positions[itest] == mpositions[ifound]

    return (; times, mtimes, mpositions)

end

function fit_sticky_target(d, ∇f!, ∇2f!, κ, pdmp_type; T =  20_000, show_progress = true, can_stick = trues(2))

    flow = pdmp_type(I(2), zeros(2))

    alg = Sticky(GridThinningStrategy(; hvp = ∇²f!), κ, can_stick)
    x0 = randn(2)
    θ0 = PDMPSamplers.initialize_velocity(flow, length(x0))
    ξ0 = SkeletonPoint(x0, θ0)

    grad = FullGradient(∇f!)
    trace0, stats0 = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)

    time_and_positions = collect(trace0)
    times = first.(time_and_positions)
    positions = getindex.(time_and_positions, 2)

    dtrace = PDMPDiscretize(trace0, 0.1)
    dtime_and_positions = collect(dtrace)
    dtimes = first.(dtime_and_positions)
    dpositions = getindex.(dtime_and_positions, 2)

    # remove duplicates
    dtimes_not_in_times = findall(t -> t ∉ times, dtimes)
    dtimes = dtimes[dtimes_not_in_times]
    dpositions = dpositions[dtimes_not_in_times]

    mtimes = [times; dtimes]
    o = sortperm(mtimes)
    mpositions = [positions; dpositions][o]
    mtimes = mtimes[o]

    itest  = 5
    ifound = findfirst(==(times[itest]), mtimes)
    @assert times[itest] == mtimes[ifound] && positions[itest] == mpositions[ifound]

    return (; times, mtimes, mpositions)

end

function setup_figure(d, mpositions, xvals = LinRange(-6f0, 6f0, 501), yvals = LinRange(-6f0, 6f0, 501))
    zvals = [pdf(d, [x, y]) for x in xvals, y in yvals ]


    f = Figure(; size = (900, 900))

    xmarginal = marginal_x.(Ref(d), xvals)
    ymarginal = marginal_y.(Ref(d), yvals)

    ax_top  = Axis(f[1, 2], limits = (nothing, 1.1.*extrema(xmarginal)))
    ax_left = Axis(f[2, 1], limits = (1.1.*extrema(ymarginal), nothing))
    ax_main = Axis(f[2, 2])
    contour!(ax_main, xvals, yvals, zvals; levels=50)
    # contourf!(ax_main, xvals, yvals, zvals; levels=20, colormap=:viridis, alpha = 0.2)
    colsize!(f.layout, 1, Relative(1//6))
    rowsize!(f.layout, 1, Relative(1//6))

    mpositions_x = getindex.(mpositions, 1)
    mpositions_y = getindex.(mpositions, 2)
    histplot_top  = hist!(ax_top,  mpositions_x[1:2]; color = (:dodgerblue, .4), normalization = :pdf)
    histplot_left = hist!(ax_left, mpositions_y[1:2]; color = (:dodgerblue, .4), normalization = :pdf, direction = :x)

    # histplot_top  = hist!(ax_top,  mpositions_x; color = (:dodgerblue, .4), normalization = :pdf)
    # histplot_left = hist!(ax_left, mpositions_y; color = (:dodgerblue, .4), normalization = :pdf, direction = :x)


    lines!(ax_left, ymarginal, xvals; color=:blue)
    lines!(ax_top, yvals, xmarginal; color=:blue)
    map((ax_top, ax_left, ax_main)) do ax
        hidedecorations!(ax, grid = false, minorgrid = false)
        # hidespines!(ax)
    end

    f, ax_main, histplot_top, histplot_left

end

function record_trace(filename, times, mtimes, mpositions, f, ax_main, histplot_top, histplot_left, no_events = 200)

    mpositions = mpositions[mtimes .< times[no_events]]
    mtimes = mtimes[mtimes .< times[no_events]]
    isevent = BitVector([t in times[1:no_events] for t in mtimes])

    pts = Point2f.(mpositions)
    mpositions_x = getindex.(mpositions, 1)
    mpositions_y = getindex.(mpositions, 2)
    record(f, filename, eachindex(mtimes); framerate = 30) do i

        i != 1 && linesegments!(ax_main, [pts[i-1], pts[i]]; color=:orange)
        if isevent[i]
            scatter!(ax_main, pts[i]; color=:red, markersize=10)
        else
            histplot_top[1]  = mpositions_x[1:min(i+1, length(mpositions_x))]
            histplot_left[1] = mpositions_y[1:min(i+1, length(mpositions_y))]
        end
    end
end

output_dir = joinpath(@__DIR__, "videos")
!isdir(output_dir) && mkpath(output_dir)


t_type = Donut; pdmp_type = ZigZag
for t_type in (Donut, Banana), pdmp_type in (ZigZag, Boomerang, BouncyParticle)

    @info "Processing target $t_type and pdmp type: $(pdmp_type)"

    d = t_type()

    name_alg = string(pdmp_type)
    name_target = string(typeof(d))
    filename = joinpath(output_dir, lowercase("$(name_alg)_$(name_target).mp4"))

    isfile(filename) && continue

    times, mtimes, mpositions = fit_target(d, pdmp_type; T = 20_000, show_progress = true)
    f, ax_main, histplot_top, histplot_left = setup_figure(d, mpositions)
    record_trace(filename, times, mtimes, mpositions, f, ax_main, histplot_top, histplot_left)

end

t_type0 = SpikeAndSlabDist{Bernoulli, ZeroMeanIsoNormal}
D, κ, ∇f!, ∇²f!, ∂fxᵢ = gen_data2(t_type0, 2, [.8, .9])
Distributions.pdf(d::SpikeAndSlabDist, x) = pdf(d.slab_dist, x)
marginal_x(d::SpikeAndSlabDist, x) = pdf(Normal(), x) * d.spike_dist.v[1].p
marginal_y(d::SpikeAndSlabDist, x) = pdf(Normal(), x) * d.spike_dist.v[2].p
times, mtimes, mpositions = fit_sticky_target(D, ∇f!, ∇²f!, κ, BouncyParticle, T = 20_000, show_progress = true)
f, ax_main, histplot_top, histplot_left = setup_figure(D, mpositions)
pdmp_type = BouncyParticle
name_alg = string(pdmp_type)
name_target = string(t_type0)
filename = joinpath(output_dir, lowercase("$(name_alg)_$(name_target).mp4"))
record_trace(filename, times, mtimes, mpositions, f, ax_main, histplot_top, histplot_left)


prob = [.8, .9]
@model function ss_normal(prob)
    γ ~ arraydist([Bernoulli(p) for p in prob])
    γ_int = Int.(γ)
    sγ = sum(γ_int)
    x ~ SpikeAndSlabPrior(γ_int, filldist(Normal(), sγ))
end
model_ss_normal = ss_normal(prob)
dγ = Deterministic_γ_proposal()
dθ = AdaptiveRandomWalk_θ_proposal()
warmup = 1000
iter   = 1000
warmup1 = warmup ÷ 2
warmup2 = warmup - warmup1
spl_moms = MOMS(2, dγ, dθ, warmup1 = warmup1, warmup2 = warmup2, niter = iter)
spl_gibbs = Gibbs((:γ,:x) => spl_moms)
chn_moms = sample(model_ss_normal, spl_gibbs, iter, num_warmup = warmup, discard_initial = warmup)
Array(chn_moms)[:, 3]
mean(Array(chn_moms)[:, 3:4], dims = 1)
std(Array(chn_moms)[:, 3:4], dims = 1)


from = 1701
proposals = spl_moms.proposal_history[from:end, :]
is = [findlast(x->x[2] == true, spl_moms.proposal_history[1:from-1, i]) for i in 1:4]
spl_moms.proposal_history[400:402, :]
spl_moms.proposal_history[is[1], 1]
spl_moms.proposal_history[is[2], 2]
start_pos = [spl_moms.proposal_history[is[3], 3][1],
             spl_moms.proposal_history[is[4], 4][1]]

# start_pos = [spl_moms.proposal_history[1, 3][1], spl_moms.proposal_history[1, 4][1]]
filename = joinpath(output_dir, lowercase("moms_spike_and_slab2.mp4"))
f, ax_main, histplot_top, histplot_left, pts_x, pts_y = setup_figure_moms(D, start_pos)
record_trace_moms(filename, proposals, start_pos, f, ax_main, histplot_top, histplot_left, pts_x, pts_y; warmup1 = 0)

function setup_figure_moms(d, start_pos, xvals = LinRange(-6f0, 6f0, 501), yvals = LinRange(-6f0, 6f0, 501))
    zvals = [pdf(d, [x, y]) for x in xvals, y in yvals]

    f = Figure(; size = (900, 900))

    # Calculate marginals for the background
    xmarginal = marginal_x.(Ref(d), xvals)
    ymarginal = marginal_y.(Ref(d), yvals)

    ax_top  = Axis(f[1, 2], limits = (nothing, 1.1 .* extrema(xmarginal)))
    ax_left = Axis(f[2, 1], limits = (1.1 .* extrema(ymarginal), nothing))
    ax_main = Axis(f[2, 2])

    contour!(ax_main, xvals, yvals, zvals; levels=50, alpha=0.5)

    colsize!(f.layout, 1, Relative(1//6))
    rowsize!(f.layout, 1, Relative(1//6))

    # Initialize histograms with the starting position
    # We use Observables (Node) so we can push to them later without redrawing the axis
    pts_x = Observable(Float64[start_pos[1]])
    pts_y = Observable(Float64[start_pos[2]])

    histplot_top  = hist!(ax_top,  pts_x; color = (:dodgerblue, .4), normalization = :pdf)
    histplot_left = hist!(ax_left, pts_y; color = (:dodgerblue, .4), normalization = :pdf, direction = :x)

    lines!(ax_left, ymarginal, xvals; color=:blue)
    lines!(ax_top, yvals, xmarginal; color=:blue)

    map((ax_top, ax_left, ax_main)) do ax
        hidedecorations!(ax, grid = false, minorgrid = false)
    end

    return f, ax_main, histplot_top, histplot_left, pts_x, pts_y
end

function record_trace_moms(filename, proposals, start_pos, f, ax_main, histplot_top, histplot_left, pts_x, pts_y; warmup1 = 0)

    # We expand the loop to visualize every column of the scan individually
    n_rows = size(proposals, 1)
    n_total_steps = n_rows * 4

    # -- Visual Element Setup --
    current_pt = Observable(Point2f(start_pos))
    proposal_pt = Observable(Point2f(start_pos))

    arrow_pos = Observable([Point2f(start_pos)])
    arrow_uv  = Observable([Point2f(0, 0)])
    arrow_col = Observable{Any}(:transparent) # Start invisible

    trace_lines = lines!(ax_main, [Point2f(start_pos)], color = :orange, alpha = 0.5)
    # arrows2d!(ax_main, arrow_pos, arrow_uv, color = arrow_col, shaftwidth = 2, tipsize = 5)
    arrows2d!(ax_main, arrow_pos, arrow_uv, color = arrow_col, shaftwidth = 2, tipwidth = 5)
    scatter!(ax_main, current_pt, color=:black, markersize=12)
    scatter!(ax_main, proposal_pt, color=arrow_col, markersize=8)

    history = Point2f[start_pos]

    # Helper: Detect if a tuple entry is real data or uninitialized memory
    function is_real_proposal(val, accepted)
        # 1. If accepted is true, it is ALWAYS a real move
        accepted && return true

        # 2. If rejected, it's real only if the value is a 'normal' float
        # (excludes NaNs, Infs, and the tiny e-300 denormals common in uninit memory)
        return !isnan(val) && abs(val) > 1e-100
    end

    record(f, filename, 1:n_total_steps; framerate = 30) do step_idx

        # Calculate which row and column we are processing
        # step_idx 1 -> row 1, col 1
        # step_idx 2 -> row 1, col 2
        i   = (step_idx - 1) ÷ 4 + 1
        col = (step_idx - 1) % 4 + 1

        state = history[end]
        cx, cy = state[1], state[2]

        # Skip processing if we are out of bounds (safety)
        if i > n_rows
            return
        end

        val, accepted = proposals[i, col]

        # -- Step 1: Check if this column contains valid data --
        if !is_real_proposal(val, accepted)
            # It's memory garbage. Hide arrow, stay put.
            arrow_col[] = :transparent
            return
        end

        # -- Step 2: Warmup checks (optional) --
        # If we are in warmup and this is a "Death" move (col 1 or 2), we might skip showing it
        if i < warmup1 && (col == 1 || col == 2)
            arrow_col[] = :transparent
            return
        end

        # -- Step 3: Calculate Proposal --
        # Logic:
        # Col 1: Propose x -> 0
        # Col 2: Propose y -> 0
        # Col 3: Propose x -> val (Update or Birth)
        # Col 4: Propose y -> val (Update or Birth)

        px, py = cx, cy

        if col == 1
            px = 0.0
        elseif col == 2
            py = 0.0
        elseif col == 3
            px = val
        elseif col == 4
            py = val
        end

        prop = Point2f(px, py)

        # -- Step 4: Update Visuals --
        current_pt[] = state

        # Only show arrow if the proposal is actually different from current state
        # (Visual clutter reduction: don't show arrow length 0)
        if norm(prop - state) > 1e-6
            arrow_pos[] = [state]
            arrow_uv[]  = [prop - state]
            proposal_pt[] = prop

            status_color = accepted ? :green : :red
            arrow_col[] = status_color
        else
            # Proposal equals current state (e.g. proposing 0 when already at 0)
            # Just flash the dot color
            arrow_uv[] = [Point2f(0,0)]
            proposal_pt[] = prop
            arrow_col[] = accepted ? :green : :red
        end

        # -- Step 5: Update History (Logic) --
        if accepted
            push!(history, prop)

            # Update trace line
            trace_lines[1] = history

            # Update Histograms
            push!(pts_x[], px)
            push!(pts_y[], py)
        else
            push!(history, state)
            push!(pts_x[], cx)
            push!(pts_y[], cy)
        end

        # Update histograms every few frames to save performance, or every frame
        notify(pts_x)
        notify(pts_y)
    end
end

function record_trace_moms(filename, proposals, start_pos, f, ax_main, histplot_top, histplot_left, pts_x, pts_y; warmup1 = 0)

    n_iter = size(proposals, 1) - 1

    # -- Visual Element Setup --
    current_pt = Observable(Point2f(start_pos))
    proposal_pt = Observable(Point2f(start_pos))

    arrow_pos = Observable([Point2f(start_pos)])
    arrow_uv  = Observable([Point2f(0, 0)])
    arrow_col = Observable{Any}(:gray)

    trace_lines = lines!(ax_main, [Point2f(start_pos)], color = :orange, alpha = 0.5)
    arrows2d!(ax_main, arrow_pos, arrow_uv, color = arrow_col, shaftwidth = 2, tipwidth = 5)
    scatter!(ax_main, current_pt, color=:black, markersize=12)
    scatter!(ax_main, proposal_pt, color=arrow_col, markersize=8)

    history = Point2f[start_pos]

    # Helper to detect if a tuple entry is likely just memory garbage
    # We assume valid proposals have a value > 1e-100 (or < -1e-100)
    # We treat exact 0.0 as "empty" since it's the default zero-init
    function is_real_proposal(val, accepted)
        # If it's accepted, it's definitely real (boolean flag is strong signal)
        !isnan(val) && accepted && return true

        # If rejected, check if the float looks like valid data
        # (not NaN, not denormal small, not zero-init)
        return !isnan(val) && abs(val) > 1e-100
    end

    record(f, filename, 1:n_iter; framerate = 10) do i

        state = history[end]
        cx, cy = state[1], state[2]

        active_col = nothing

        # Scan all 4 columns to find the "real" move for this row
        # We prioritize 'Accepted' moves if multiple exist (rare)
        for col in 1:4
            val, accepted = proposals[i, col]

            if is_real_proposal(val, accepted)

                # Apply warmup/zero constraints
                if i < warmup1 && (col == 1 || col == 2)
                    continue
                end
                if col == 3 && cx == 0.0
                    continue
                end
                if col == 4 && cy == 0.0
                    continue
                end

                active_col = col
                # If we found an accepted move, stop looking.
                # Otherwise keep looking in case a later column is the 'real' one
                if accepted
                    break
                end
            end
        end

        if isnothing(active_col)
            # No valid move found in this row -> stay put, hide arrow
            push!(history, state)
            push!(pts_x[], cx)
            push!(pts_y[], cy)
            arrow_uv[] = [Point2f(0, 0)]
            arrow_col[] = :transparent
            return
        end

        # -- Calculate Proposal --
        (val, accepted) = proposals[i, active_col]
        px, py = cx, cy

        # if accepted
        if active_col == 1
            px = 0.0
        elseif active_col == 2
            py = 0.0
        elseif active_col == 3
            px = val
        elseif active_col == 4
            py = val
        end
        # end

        prop = Point2f(px, py)

        # -- Update Visuals --
        current_pt[] = state
        arrow_pos[] = [state]
        arrow_uv[]  = [prop - state]
        proposal_pt[] = prop

        status_color = accepted ? :green : :red
        arrow_col[] = status_color

        # -- Update History --
        if accepted
            push!(history, prop)
            trace_lines[1] = history
            push!(pts_x[], px)
            push!(pts_y[], py)
        else
            push!(history, state)
            push!(pts_x[], cx)
            push!(pts_y[], cy)
        end

        notify(pts_x)
        notify(pts_y)
    end
end

d = Donut()#Donut(3.0, 1.0)
# d = Banana()

xvals = LinRange(-6f0, 6f0, 501)
yvals = LinRange(-6f0, 6f0, 501)
zvals = [pdf(d, [x, y]) for x in xvals, y in yvals ]
# ymarginal = vec(sum(zvals, dims=1))
# ymarginal ./= sum(ymarginal)

# xmarginal = vec(sum(zvals, dims=2))
# xmarginal ./= sum(xmarginal)

xmarginal = marginal_x.(Ref(d), xvals)
ymarginal = marginal_y.(Ref(d), yvals)

testval_x = [1.124, -.876]
testval_v = [0.1, 0.2]

@assert DI.gradient(x -> logpdf(d, x), DI.AutoForwardDiff(), testval_x) ≈ gradlogpdf(d, testval_x)
@assert DI.hessian( x -> logpdf(d, x), DI.AutoForwardDiff(), testval_x) * testval_v ≈ hessian_vector_product(d, testval_x, testval_v)
# would be nicer but does not work with ForwardDiff?
# DI.hvp(x -> logpdf(d, x), DI.AutoForwardDiff(), testval_x, testval_v)

# ForwardDiff.gradient(x -> logpdf(d, x), testval_x) ≈ gradlogpdf(d, testval_x)
# ForwardDiff.hessian( x -> logpdf(d, x), testval_x) * testval_v ≈ hessian_vector_product(d, testval_x, testval_v)

pdmp_type = ZigZag
show_progress = true

∇f!(out, x) = out .= .-gradlogpdf(d, x)
∇²f!(out, x, v) = out .= .-hessian_vector_product(d, x, v)


flow = pdmp_type(I(2), zeros(2))
alg = GridThinningStrategy(; hvp = ∇²f!)
x0 = randn(2)
θ0 = PDMPSamplers.initialize_velocity(flow, length(x0))
ξ0 = SkeletonPoint(x0, θ0)

grad = FullGradient(∇f!)
trace0, stats0 = pdmp_sample(ξ0, flow, grad, GridThinningStrategy(; hvp = ∇²f!), 0.0, 20000, progress=show_progress)

time_and_positions = collect(trace0)
times = first.(time_and_positions)
positions = getindex.(time_and_positions, 2)

dtrace = PDMPDiscretize(trace0, 0.1)
dtime_and_positions = collect(dtrace)
dtimes = first.(dtime_and_positions)
dpositions = getindex.(dtime_and_positions, 2)

# remove duplicates
dtimes_not_in_times = findall(t -> t ∉ times, dtimes)
dtimes = dtimes[dtimes_not_in_times]
dpositions = dpositions[dtimes_not_in_times]

mtimes = [times; dtimes]
o = sortperm(mtimes)
mpositions = [positions; dpositions][o]
mtimes = mtimes[o]

itest  = 5
ifound = findfirst(==(times[itest]), mtimes)
@assert times[itest] == mtimes[ifound] && positions[itest] == mpositions[ifound]

no_events = 300
mpositions = mpositions[mtimes .< times[no_events]]
mtimes = mtimes[mtimes .< times[no_events]]
isevent = BitVector([t in times[1:no_events] for t in mtimes])


frames = length(mtimes)

mpositions_x = getindex.(mpositions, 1)
mpositions_y = getindex.(mpositions, 2)

f = Figure()
ax_top  = Axis(f[1, 2], limits = (nothing, 1.1.*extrema(xmarginal)))
ax_left = Axis(f[2, 1], limits = (1.1.*extrema(ymarginal), nothing))
ax_main = Axis(f[2, 2])
contour!(ax_main, xvals, yvals, zvals; levels=50)
# contourf!(ax_main, xvals, yvals, zvals; levels=20, colormap=:viridis, alpha = 0.2)
colsize!(f.layout, 1, Relative(1//6))
rowsize!(f.layout, 1, Relative(1//6))

# histplot_top  = hist!(ax_top,  mpositions_x[1:2]; color = (:dodgerblue, .4), normalization = :pdf)
# histplot_left = hist!(ax_left, mpositions_y[1:2]; color = (:dodgerblue, .4), normalization = :pdf, direction = :x)

histplot_top  = hist!(ax_top,  mpositions_x; color = (:dodgerblue, .4), normalization = :pdf)
histplot_left = hist!(ax_left, mpositions_y; color = (:dodgerblue, .4), normalization = :pdf, direction = :x)


lines!(ax_left, ymarginal, xvals; color=:blue)
lines!(ax_top, yvals, xmarginal; color=:blue)
map((ax_top, ax_left, ax_main)) do ax
    hidedecorations!(ax, grid = false, minorgrid = false)
    # hidespines!(ax)
end

f

pts = Point2f.(mpositions)



name_alg = string(pdmp_type)
name_target = string(typeof(d))

!isdir(output_dir) && mkpath(output_dir)
filename = joinpath(output_dir, lowercase("$(name_alg)_$(name_target)2.mp4"))
record(f, filename, eachindex(mtimes);
        framerate = 30) do i

    i != 1 && linesegments!(ax_main, [pts[i-1], pts[i]]; color=:orange)
    if isevent[i]
        scatter!(ax_main, pts[i]; color=:red, markersize=10)
    else
        histplot_top[1]  = mpositions_x[1:min(i+1, length(mpositions_x))]
        histplot_left[1] = mpositions_y[1:min(i+1, length(mpositions_y))]
    end
end


# Generate some data that changes over time
frames = 100
data = randn(1000)

# Set up figure and histogram
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1]; title = "Animated Histogram")
histplot = hist!(ax, data[1:1]; bins = -5:0.25:5, color = :dodgerblue, normalization = :pdf)
lines!(ax, -5..5, x -> pdf(Normal(), x); color=:red, linewidth=2)
fig


# Animate: update the histogram each frame
record(fig, "animated_histogram.mp4", 1:frames; framerate = 30) do i
    histplot[1] = data[1:1+i]  # update the histogram data

    iszero(i % 25) && Makie.autolimits!(ax)
    ax.title = "Frame $i"
end