_reset_inner_grid!(::PoissonTimeStrategy) = nothing
_invalidate_cached_gradient!(::PoissonTimeStrategy) = nothing
_maybe_activate_constant_bound!(::PoissonTimeStrategy, ::AbstractStatisticCounter) = nothing
requires_sticky_state(::PoissonTimeStrategy) = false
_is_sticky_loop_state(::PoissonTimeStrategy) = false
