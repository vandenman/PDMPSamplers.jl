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