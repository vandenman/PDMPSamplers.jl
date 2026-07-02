
if VERSION < v"1.13.0"
    ispositive(x::Real) = x > zero(x)
    isnegative(x::Real) = x < zero(x)
else
    # code for Julia ≥ 1.13
end

pos(x) = max(zero(x), x)

# TODO: remove these!
function idot(A, j, x)
    return dot((@view A[:, j]), x)
end

struct VHVProvider{G,V,W<:Union{Nothing,AbstractVector}}
    grad::G
    vhv::V
    w_buf::W
end
VHVProvider(grad, vhv) = VHVProvider(grad, vhv, nothing)

struct FiniteDiffVHV{G}
    grad::G
    buf::Vector{Float64}
    grad_buf::Vector{Float64}
    w_buf::Vector{Float64}
end
FiniteDiffVHV(grad, buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, similar(buf), similar(buf))
FiniteDiffVHV(grad, buf::Vector{Float64}, w_buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, similar(buf), w_buf)
