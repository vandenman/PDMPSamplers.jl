
if VERSION < v"1.13.0"
    ispositive(x::Real) = x > zero(x)
    isnegative(x::Real) = x < zero(x)
else
    # code for Julia ≥ 1.13
end

pos(x) = max(zero(x), x)

# TODO: these two need to go, actually, do they?
normsq(x::Real)           = abs2(x)
normsq(x::AbstractVector) = sum(abs2, x)

mutable struct StatisticCounter
    reflections_events::Int
    reflections_accepted::Int
    refreshment_events::Int
    sticky_events::Int
    ∇f_calls::Int
    ∇²f_calls::Int
    last_rejected::Bool
    grid_builds::Int
    grid_shrinks::Int
    grid_grows::Int
    grid_early_stops::Int
    grid_points_evaluated::Int
    grid_points_skipped::Int
    grid_N_current::Int
    elapsed_time::Float64
end
StatisticCounter() = StatisticCounter(0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0.0)



mutable struct HealthMonitor
    consecutive_rejects::Int
    const limit::Int
    HealthMonitor(; consecutive_reject_limit=1000) = new(0, consecutive_reject_limit)
end

function check_health!(monitor::HealthMonitor, stats::StatisticCounter)
    if stats.last_rejected
        monitor.consecutive_rejects += 1
    else
        monitor.consecutive_rejects = 0
    end

    if monitor.consecutive_rejects > monitor.limit
        # error("Stuck! Rejected $(monitor.limit) consecutive moves.")
        error("The algorithm rejected $(monitor.limit) consecutive proposals. Check the algorithm and model settings.")
    end
end

# TODO: remove these!
function idot(A, j, x)
    return dot((@view A[:, j]), x)
end

# function idot(A::SparseMatrixCSC, j, x)
#     rows = rowvals(A)
#     vals = nonzeros(A)
#     s = zero(eltype(x))
#     # @inbounds
#     for i in nzrange(A, j)
#         s += vals[i]'*x[rows[i]]
#     end
#     s
# end
