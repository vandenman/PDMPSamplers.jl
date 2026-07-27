# Internal helper callables used by generated precompile signatures.
# They keep src/precompile_statements.jl independent of test-local closures.
function _precompile_neg_gradient!(out::AbstractVector, x::AbstractVector)
    copyto!(out, x)
    return out
end

function _precompile_neg_hvp!(out::AbstractVector, x::AbstractVector, v::AbstractVector)
    copyto!(out, v)
    return out
end

function _precompile_model(d::Integer)
    return PDMPModel(Int(d), FullGradient(_precompile_neg_gradient!), _precompile_neg_hvp!)
end

const _PRECOMPILE_STATEMENTS_PATH = joinpath(@__DIR__, "precompile_statements.jl")
if get(ENV, "PDMPSAMPLERS_DISABLE_GENERATED_PRECOMPILE", "false") != "true" &&
        isfile(_PRECOMPILE_STATEMENTS_PATH)
    include(_PRECOMPILE_STATEMENTS_PATH)
end
