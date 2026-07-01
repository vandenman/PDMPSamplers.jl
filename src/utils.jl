
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
