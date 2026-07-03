#=

Statistic counters

These measure counts within the sampler itself.

The code below is a llm-generated metaprogrammed drop-in replacement for
the hand-written hot-garbage counter plumbing that used to be there.

They are very useful for debugging and performance analysis.
But most users don't need them, so the current design is that
they are parametric so that end-users won't even notice they exist,
assuming Julia compiles away the no-op methods.


Public API preserved:
    - AbstractStatisticCounter
    - MultiCounter
    - StatisticCounter(), DevelStatisticCounter()
    - _inc_counter_XXX(...)
    - _set_counter_XXX(...)
    - _get_counter_XXX(...)

The counter structs remain explicit. Constructors and counter operation methods are generated.
=#


abstract type AbstractStatisticCounter end

# MultiCounter tuple recursion — fully unrolled at compile time for concrete tuple types.
@inline _apply_to_all(f, ::Tuple{}, args...) = nothing
@inline function _apply_to_all(f, t::Tuple, args...)
    f(t[1], args...)
    _apply_to_all(f, Base.tail(t), args...)
    nothing
end

@inline _apply_sum(f, ::Tuple{}) = 0
@inline function _apply_sum(f, t::Tuple)
    f(t[1]) + _apply_sum(f, Base.tail(t))
end

@inline _apply_sum_float(f, ::Tuple{}) = 0.0
@inline function _apply_sum_float(f, t::Tuple)
    f(t[1]) + _apply_sum_float(f, Base.tail(t))
end

@inline _apply_any(f, ::Tuple{}) = false
@inline function _apply_any(f, t::Tuple)
    f(t[1]) || _apply_any(f, Base.tail(t))
end

# MultiCounter: parametric composition of counters
struct MultiCounter{C<:Tuple} <: AbstractStatisticCounter
    counters::C
end

function _counter_getproperty(counters::Tuple, name::Symbol)
    for counter in counters
        hasfield(typeof(counter), name) && return getfield(counter, name)
    end
    throw(ErrorException("counter field $(name) is not present in this MultiCounter"))
end

function _counter_setproperty!(counters::Tuple, name::Symbol, value)
    for counter in counters
        if hasfield(typeof(counter), name)
            setfield!(counter, name, value)
            return value
        end
    end
    throw(ErrorException("counter field $(name) is not present in this MultiCounter"))
end

function Base.getproperty(m::MultiCounter, name::Symbol)
    name === :counters && return getfield(m, :counters)
    return _counter_getproperty(getfield(m, :counters), name)
end

function Base.setproperty!(m::MultiCounter, name::Symbol, value)
    name === :counters && throw(ErrorException("cannot replace MultiCounter.counters"))
    return _counter_setproperty!(getfield(m, :counters), name, value)
end

function Base.propertynames(m::MultiCounter, private::Bool=false)
    names = Symbol[:counters]
    for counter in getfield(m, :counters)
        append!(names, fieldnames(typeof(counter)))
    end
    return tuple(unique(names)...)
end

mutable struct HealthMonitor
    consecutive_rejects::Int
    const limit::Int
    HealthMonitor(; consecutive_reject_limit=1000) = new(0, consecutive_reject_limit)
end

function check_health!(monitor::HealthMonitor, stats::AbstractStatisticCounter)
    if _get_counter_last_rejected(stats)
        monitor.consecutive_rejects += 1
    else
        monitor.consecutive_rejects = 0
    end

    if monitor.consecutive_rejects > monitor.limit
        error("The algorithm rejected $(monitor.limit) consecutive proposals. Check the algorithm and model settings.")
    end
end

# Zero/default construction helpers
_counter_default(::Type{T}) where {T} = zero(T)
_counter_default(::Type{Bool}) = false
_counter_default(::Type{Symbol}) = :none

@inline _zero_counter(::Type{T}) where {T<:AbstractStatisticCounter} = T(ntuple(i -> _counter_default(fieldtype(T, i)), Val(fieldcount(T)))...)

function _counter_struct_name(ex)
    if ex isa Symbol
        return ex
    elseif ex isa Expr && ex.head === :(<:)
        return _counter_struct_name(ex.args[1])
    elseif ex isa Expr && ex.head === :curly
        return _counter_struct_name(ex.args[1])
    else
        error("Could not extract struct name from: $ex")
    end
end

"""
    @counter_struct mutable struct MyCounter <: AbstractStatisticCounter
        field_a::Int
        field_b::Float64
    end

Define the struct and a zero/default constructor `MyCounter()`.
"""
macro counter_struct(def)
    def isa Expr && def.head === :struct ||
        error("@counter_struct must wrap a struct definition")

    T = _counter_struct_name(def.args[2])

    return esc(quote
        $def
        $T() = _zero_counter($T)
    end)
end

# Operation generation helpers
function _gen_inc(T, name)
    fname = Symbol("_inc_counter_", name)
    qname = QuoteNode(name)

    quote
        @inline $(fname)(::AbstractStatisticCounter) = nothing

        @inline function $(fname)(c::$(T))
            setfield!(c, $(qname), getfield(c, $(qname)) + 1)
            nothing
        end

        @inline function $(fname)(m::MultiCounter)
            _apply_to_all($(fname), m.counters)
            nothing
        end
    end
end

function _gen_incval(T, name)
    fname = Symbol("_inc_counter_", name)
    qname = QuoteNode(name)

    quote
        @inline $(fname)(::AbstractStatisticCounter, val) = nothing

        @inline function $(fname)(c::$(T), val)
            setfield!(c, $(qname), getfield(c, $(qname)) + val)
            nothing
        end

        @inline function $(fname)(m::MultiCounter, val)
            _apply_to_all($(fname), m.counters, val)
            nothing
        end
    end
end

function _gen_set(T, name)
    fname = Symbol("_set_counter_", name)
    qname = QuoteNode(name)

    quote
        @inline $(fname)(::AbstractStatisticCounter, val) = nothing

        @inline function $(fname)(c::$(T), val)
            setfield!(c, $(qname), val)
            nothing
        end

        @inline function $(fname)(m::MultiCounter, val)
            _apply_to_all($(fname), m.counters, val)
            nothing
        end
    end
end

function _gen_get_sum(T, name)
    fname = Symbol("_get_counter_", name)
    qname = QuoteNode(name)

    quote
        @inline $(fname)(::AbstractStatisticCounter) = 0

        @inline $(fname)(c::$(T)) = getfield(c, $(qname))

        @inline function $(fname)(m::MultiCounter)
            _apply_sum($(fname), m.counters)
        end
    end
end

function _gen_get_float(T, name)
    fname = Symbol("_get_counter_", name)
    qname = QuoteNode(name)

    quote
        @inline $(fname)(::AbstractStatisticCounter) = 0.0

        @inline $(fname)(c::$(T)) = getfield(c, $(qname))

        @inline function $(fname)(m::MultiCounter)
            _apply_sum_float($(fname), m.counters)
        end
    end
end

function _gen_get_any(T, name)
    fname = Symbol("_get_counter_", name)
    qname = QuoteNode(name)

    quote
        @inline $(fname)(::AbstractStatisticCounter) = false

        @inline $(fname)(c::$(T)) = getfield(c, $(qname))

        @inline function $(fname)(m::MultiCounter)
            _apply_any($(fname), m.counters)
        end
    end
end

"""
    @counter_ops MyCounter begin
        inc(field_a, field_b)
        incval(accumulator_field)
        set(state_field)
        get_sum(field_a)
        get_float(accumulator_field)
        get_any(bool_field)
    end

Generated operations:
  - inc(name):      `_inc_counter_name(c)` adds 1.
  - incval(name):   `_inc_counter_name(c, val)` adds `val`.
  - set(name):      `_set_counter_name(c, val)` assigns `val`.
  - get_sum(name):  `_get_counter_name(c)` returns/sums integer-like values.
  - get_float(name): `_get_counter_name(c)` returns/sums float-like values.
  - get_any(name):  `_get_counter_name(c)` returns OR across counters.
"""
macro counter_ops(T, block)
    out = Expr(:block)

    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr && stmt.head === :call ||
            error("@counter_ops expects calls like inc(a, b), incval(a), set(a), get_sum(a)")

        op = stmt.args[1]
        names = stmt.args[2:end]

        generator =
            op === :inc       ? _gen_inc :
            op === :incval    ? _gen_incval :
            op === :set       ? _gen_set :
            op === :get_sum   ? _gen_get_sum :
            op === :get_float ? _gen_get_float :
            op === :get_any   ? _gen_get_any :
            error("Unknown counter operation: $op")

        for name in names
            name isa Symbol || error("Expected field name symbol, got: $name")
            push!(out.args, generator(T, name))
        end
    end

    return esc(out)
end

"""
    @counter_bundle Alias CounterA CounterB CounterC

Define

    const Alias = MultiCounter{Tuple{CounterA,CounterB,CounterC}}
    Alias() = MultiCounter((CounterA(), CounterB(), CounterC()))
"""
macro counter_bundle(alias, Ts...)
    tuple_type = Expr(:curly, :Tuple, Ts...)
    calls = [:( $(T)() ) for T in Ts]

    return esc(quote
        const $(alias) = MultiCounter{$(tuple_type)}

        function $(alias)()
            MultiCounter(($(calls...),))
        end
    end)
end


# Concrete counter types — grouped by domain
@counter_struct mutable struct BasicEventCounter <: AbstractStatisticCounter
    reflections_events::Int
    reflections_accepted::Int
    refreshment_events::Int
    sticky_events::Int
    boundary_reflections::Int
    last_rejected::Bool
end

@counter_ops BasicEventCounter begin
    inc(
        reflections_events,
        reflections_accepted,
        refreshment_events,
        sticky_events,
        boundary_reflections,
    )

    set(last_rejected)

    get_sum(
        reflections_events,
        reflections_accepted,
        refreshment_events,
        sticky_events,
    )

    get_any(last_rejected)
end

@counter_struct mutable struct SupportBoundaryCounter <: AbstractStatisticCounter
    support_boundary_events::Int
    support_boundary_refresh_attempts::Int
    support_boundary_refresh_failures::Int
end

@counter_ops SupportBoundaryCounter begin
    inc(
        support_boundary_events,
        support_boundary_refresh_attempts,
        support_boundary_refresh_failures,
    )
end

@counter_struct mutable struct GradientCallCounter <: AbstractStatisticCounter
    ∇f_calls::Int
    ∇²f_calls::Int
end

@counter_ops GradientCallCounter begin
    inc(
        ∇f_calls,
        ∇²f_calls,
    )

    get_sum(
        ∇f_calls,
        ∇²f_calls,
    )
end

@counter_struct mutable struct GridThinningCounter <: AbstractStatisticCounter
    grid_builds::Int
    grid_shrinks::Int
    grid_grows::Int
    grid_early_stops::Int
    grid_points_evaluated::Int
    grid_points_skipped::Int
    grid_N_current::Int
    grid_horizon_hits::Int
    grid_schedule_samples::Int
    grid_N_sum::Float64
    grid_tmax_sum::Float64
    grid_h_sum::Float64
    grid_certificate_calls::Int
    grid_certificate_fallbacks::Int
    grid_budget_extensions::Int
    grid_budget_cells_built::Int
    grid_budget_area_built::Float64
    grid_budget_exponential_sum::Float64
    grid_budget_tail_restarts::Int
    grid_endpoint_evaluations::Int
    grid_endpoint_derivative_calls::Int
    grid_endpoint_gradient_calls::Int
    grid_endpoint_hessian_calls::Int
    grid_cached_endpoint_reuses::Int
    grid_acceptance_tests::Int
    grid_acceptance_gradient_calls::Int
    grid_bound_violations::Int
    grid_endpoint_derivative_points_loaded::Int
    grid_resets_from_dynamics_adaptation::Int
end

@counter_ops GridThinningCounter begin
    inc(
        grid_builds,
        grid_shrinks,
        grid_grows,
        grid_early_stops,
        grid_horizon_hits,
        grid_schedule_samples,
        grid_certificate_calls,
        grid_budget_extensions,
        grid_budget_tail_restarts,
        grid_endpoint_evaluations,
        grid_endpoint_derivative_calls,
        grid_endpoint_gradient_calls,
        grid_endpoint_hessian_calls,
        grid_cached_endpoint_reuses,
        grid_acceptance_tests,
        grid_acceptance_gradient_calls,
        grid_bound_violations,
        grid_resets_from_dynamics_adaptation,
    )

    incval(
        grid_points_evaluated,
        grid_points_skipped,
        grid_N_sum,
        grid_tmax_sum,
        grid_h_sum,
        grid_certificate_fallbacks,
        grid_budget_cells_built,
        grid_budget_area_built,
        grid_budget_exponential_sum,
        grid_endpoint_derivative_points_loaded,
    )

    set(grid_N_current)

    get_sum(
        grid_acceptance_tests,
    )
end

@counter_struct mutable struct ConstantBoundCounter <: AbstractStatisticCounter
    constant_bound_attempts::Int
    constant_bound_accepts::Int
    constant_bound_rejections::Int
    constant_bound_violations::Int
    constant_bound_safety_fallbacks::Int
end

@counter_ops ConstantBoundCounter begin
    inc(
        constant_bound_attempts,
        constant_bound_accepts,
        constant_bound_rejections,
        constant_bound_violations,
        constant_bound_safety_fallbacks,
    )
end

@counter_struct mutable struct StickyStatsCounter <: AbstractStatisticCounter
    sticky_inner_searches::Int
    sticky_inner_wins::Int
    sticky_inner_wasted_by_sticky::Int
    sticky_inner_wasted_by_refresh::Int
    sticky_all_frozen_events::Int
end

@counter_ops StickyStatsCounter begin
    inc(
        sticky_inner_searches,
        sticky_inner_wins,
        sticky_inner_wasted_by_sticky,
        sticky_inner_wasted_by_refresh,
        sticky_all_frozen_events,
    )
end

@counter_struct mutable struct AffineBoundCounter <: AbstractStatisticCounter
    affine_roof_cells::Int
    affine_inflated_cells::Int
    affine_constant_cells::Int
    affine_area_constant_equiv::Float64
    affine_area_hybrid::Float64
    affine_area_saved::Float64
    affine_cells_skipped_by_min_gain::Int
    affine_segments_added::Int
    affine_bound_violations::Int
end

@counter_ops AffineBoundCounter begin
    inc(
        affine_roof_cells,
        affine_inflated_cells,
        affine_constant_cells,
        affine_cells_skipped_by_min_gain,
        affine_bound_violations,
    )

    incval(
        affine_area_constant_equiv,
        affine_area_hybrid,
        affine_area_saved,
        affine_segments_added,
    )

    get_float(affine_area_constant_equiv)
end

@counter_struct mutable struct CertifiedAutoCounter <: AbstractStatisticCounter
    auto_flat_cells::Int
    auto_affine_cells::Int
    auto_area_saved::Float64
    auto_affine_fraction::Float64
    auto_used_affine_grids::Int
    auto_used_flat_grids::Int
end

@counter_ops CertifiedAutoCounter begin
    inc(
        auto_flat_cells,
        auto_affine_cells,
        auto_used_affine_grids,
        auto_used_flat_grids,
    )

    incval(auto_area_saved)

    set(auto_affine_fraction)

    get_sum(
        auto_flat_cells,
        auto_affine_cells,
    )

    get_float(auto_area_saved)
end

mutable struct ComponentwiseAffineCounter <: AbstractStatisticCounter
    # Cumulative channel-work counters, not model dimension fields.
    componentwise_channels::Int
    componentwise_channel_point_evaluations::Int
    componentwise_affine_cells::Int
    componentwise_flat_fallback_cells::Int
    componentwise_flat_fallback_segment_cap::Int
    componentwise_flat_fallback_numerical::Int
    componentwise_affine_skipped_by_area_gate::Int
    componentwise_affine_skipped_by_policy::Int
    componentwise_flat_fallback_area_gate::Int
    componentwise_flat_fallback_policy::Int
    componentwise_flat_fallback_other::Int
    componentwise_affine_segments_added::Int
    componentwise_breakpoints_merged::Int
    componentwise_area_saved::Float64
    componentwise_proposed_breakpoints_per_cell::Vector{Float64}
    componentwise_segments_per_cell::Vector{Float64}
    componentwise_zero_crossings_per_cell::Vector{Float64}
    componentwise_area_saved_per_cell::Vector{Float64}
    componentwise_area_saved_fraction_per_cell::Vector{Float64}
end

function ComponentwiseAffineCounter()
    return ComponentwiseAffineCounter(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0,
        Float64[], Float64[], Float64[], Float64[], Float64[],
    )
end

@counter_ops ComponentwiseAffineCounter begin
    inc(
        componentwise_affine_cells,
        componentwise_flat_fallback_cells,
        componentwise_flat_fallback_segment_cap,
        componentwise_flat_fallback_numerical,
        componentwise_affine_skipped_by_area_gate,
        componentwise_affine_skipped_by_policy,
        componentwise_flat_fallback_area_gate,
        componentwise_flat_fallback_policy,
        componentwise_flat_fallback_other,
    )

    incval(
        componentwise_channels,
        componentwise_channel_point_evaluations,
        componentwise_affine_segments_added,
        componentwise_breakpoints_merged,
        componentwise_area_saved,
    )

    get_sum(
        componentwise_channels,
        componentwise_channel_point_evaluations,
        componentwise_affine_cells,
        componentwise_flat_fallback_cells,
        componentwise_flat_fallback_segment_cap,
        componentwise_flat_fallback_numerical,
        componentwise_affine_skipped_by_area_gate,
        componentwise_affine_skipped_by_policy,
        componentwise_flat_fallback_area_gate,
        componentwise_flat_fallback_policy,
        componentwise_flat_fallback_other,
        componentwise_affine_segments_added,
        componentwise_breakpoints_merged,
    )

    get_float(componentwise_area_saved)
end

@inline _record_counter_componentwise_cell_diagnostics!(::AbstractStatisticCounter, args...) = nothing
@inline _record_counter_componentwise_cell_diagnostics!(::Nothing, args...) = nothing

@inline function _record_counter_componentwise_cell_diagnostics!(
    c::ComponentwiseAffineCounter,
    proposed_breakpoints,
    segments,
    zero_crossings,
    area_saved,
    area_saved_fraction,
)
    push!(c.componentwise_proposed_breakpoints_per_cell, Float64(proposed_breakpoints))
    push!(c.componentwise_segments_per_cell, Float64(segments))
    push!(c.componentwise_zero_crossings_per_cell, Float64(zero_crossings))
    push!(c.componentwise_area_saved_per_cell, Float64(area_saved))
    push!(c.componentwise_area_saved_fraction_per_cell, Float64(area_saved_fraction))
    return nothing
end

@inline function _record_counter_componentwise_cell_diagnostics!(
    m::MultiCounter,
    args...,
)
    _apply_to_all(_record_counter_componentwise_cell_diagnostics!, m.counters, args...)
    return nothing
end

@counter_struct mutable struct PhaseSummaryCounter <: AbstractStatisticCounter
    warmup_events::Int
    main_events::Int
    warmup_gradient_calls::Int
    main_gradient_calls::Int
    warmup_hessian_calls::Int
    main_hessian_calls::Int
    warmup_elapsed_time::Float64
    main_elapsed_time::Float64
    elapsed_time::Float64
    stop_reason::Symbol
end

@counter_ops PhaseSummaryCounter begin
    incval(
        warmup_events,
        main_events,
        warmup_gradient_calls,
        main_gradient_calls,
        warmup_hessian_calls,
        main_hessian_calls,
        warmup_elapsed_time,
        main_elapsed_time,
    )

    set(
        elapsed_time,
        stop_reason,
    )
end

@counter_struct mutable struct LazyBoundCounter <: AbstractStatisticCounter
    lazy_fallback_low_tightness::Int
    lazy_fallback_bound_violation::Int
    lazy_proposal_attempts::Int
    lazy_proposal_rejections::Int
end

@counter_ops LazyBoundCounter begin
    inc(
        lazy_fallback_low_tightness,
        lazy_fallback_bound_violation,
    )

    incval(
        lazy_proposal_attempts,
        lazy_proposal_rejections,
    )
end

# ===========================================================================================
# Convenience type aliases for backward-compatible construction
# ===========================================================================================

@counter_bundle(
    StatisticCounter,
    BasicEventCounter,
    SupportBoundaryCounter,
    GradientCallCounter,
    GridThinningCounter,
    PhaseSummaryCounter,
    LazyBoundCounter,
)

@counter_bundle(
    DevelStatisticCounter,
    BasicEventCounter,
    SupportBoundaryCounter,
    GradientCallCounter,
    GridThinningCounter,
    ConstantBoundCounter,
    StickyStatsCounter,
    AffineBoundCounter,
    CertifiedAutoCounter,
    ComponentwiseAffineCounter,
    PhaseSummaryCounter,
    LazyBoundCounter,
)
