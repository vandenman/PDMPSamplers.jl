
if VERSION < v"1.13.0"
    ispositive(x::Real) = x > zero(x)
    isnegative(x::Real) = x < zero(x)
else
    # code for Julia ≥ 1.12
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
end
StatisticCounter() = StatisticCounter(0, 0, 0, 0, 0, 0, false)



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
