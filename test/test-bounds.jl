@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Bound Validation Tests" begin

    """
    Test that ab bounds are valid upper bounds for the rate function λ(t)
    """
    function test_rate_bounds(flow::ContinuousDynamics, ξ::SkeletonPoint,
                            ∇f!::Function, c::AbstractVector;
                            t_max::Real = 10.0, n_test_points::Int = 101,
                            test_tightness = flow isa BouncyParticle)

        # ξ = ξ0; t_max = 30.; n_test_points = 101
        grad = FullGradient(∇f!)
        cache = PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(c[1], length(ξ.x))), 0.0, ξ)
        cache = PDMPSamplers.add_gradient_to_cache(cache, ξ)


        # Get bounds
        a, b, refresh_time = PDMPSamplers.ab(ξ, c, flow, cache)

        # Test at multiple time points
        test_times = collect(range(0.0, t_max, n_test_points))

        for _ in 1:5
            # also add a few next_times
            push!(test_times, PDMPSamplers.next_time(0.0, (a, b, Inf), rand())[1])
        end

        state = PDMPSamplers.PDMPState(0.0, ξ)
        tol = .26
        for t in test_times
            # Evolve state to time t
            # ξt = move_forward_time(ξ, t, flow)
            statet = move_forward_time(state, t, flow)

            # Compute actual rate
            # ∇ϕ = compute_gradient!(ξt.x, ξt.θ, grad, flow, cache)
            ∇ϕ = compute_gradient!(statet, grad, flow, cache)
            actual_rate = λ(statet, ∇ϕ, flow)

            # Compute bound
            bound_rate = PDMPSamplers.pos(a + b * t)

            # Test that bound is valid
            @test actual_rate <= bound_rate
            # if !(actual_rate <= bound_rate)  # Small numerical tolerance
            #     error("Bound violation at t=$t: actual_rate=$actual_rate > bound_rate=$bound_rate (flow: $(typeof(flow)))")
            # end

            if test_tightness
                # Test that bound is close
                # @test actual_rate / bound_rate > 0.95
                !iszero(actual_rate) && @test isapprox(actual_rate, bound_rate, atol=tol, rtol = tol)
            end
            # if !(iszero(actual_rate) || isapprox(actual_rate, bound_rate, atol=tol, rtol = tol))
            #     error("Bound ratio violation at t=$t: actual_rate / bound_rate = $actual_rate / $bound_rate = $(actual_rate / bound_rate) (flow: $(typeof(flow)))")
            # end
        end

        # quick and dirty version to plot the above
        # actual_rates = [
        #     begin
        #         ξt = move_forward_time(ξ, t, flow)
        #         # Compute actual rate
        #         ∇ϕ = compute_gradient!(ξt.x, ξt.θ, grad, flow, cache)
        #         λ(ξt, ∇ϕ, flow)
        #     end
        #     for t in test_times
        # ]
        # bound_rates = pos.(a .+ b .* test_times)# .+ flow.λref
        # actual_rates ./ bound_rates
        # fig, _ = scatter(actual_rates, actual_rates ./ bound_rates)
        # ax2 = Axis(fig[1, 2])
        # scatter!(ax2, test_times, actual_rates, color=:blue)
        # scatter!(ax2, test_times, bound_rates, color=:orange)
        # fig


    end

    """
    Test that ab_i bounds are valid for factorized flows
    """
    function test_coordinate_bounds(flow::FactorizedDynamics, ξ::SkeletonPoint,
                                ∇ϕᵢ!::Function, c::AbstractVector;
                                t_max::Real = 10.0, n_test_points::Int = 100)

        cache = PDMPSamplers.initialize_cache(flow, CoordinateWiseGradient(∇ϕᵢ!), ThinningStrategy(LocalBounds(c)), 0.0, ξ)
        cache = PDMPSamplers.add_gradient_to_cache(cache, ξ)

        test_times = range(0.0, t_max, n_test_points)

        state = PDMPSamplers.PDMPState(0.0, ξ)
        for i in eachindex(ξ.x)
            # Get coordinate-specific bounds
            a_i, b_i = PDMPSamplers.ab_i(i, ξ, c, flow, cache)

            for t in test_times
                # Evolve state to time t
                statet = move_forward_time(state, t, flow)

                # Compute actual coordinate rate
                ∇ϕᵢ = ∇ϕᵢ!(statet.ξ.x, i)
                actual_rate_i = PDMPSamplers.λ_i(i, statet.ξ, ∇ϕᵢ, flow)

                # Compute bound
                bound_rate_i = a_i + b_i * t

                # Test that bound is valid
                if !(actual_rate_i <= bound_rate_i + 1e-10)
                    error("Coordinate bound violation at t=$t, coord=$i: actual=$actual_rate_i > bound=$bound_rate_i (flow: $(typeof(flow)))")
                end
            end
        end

        return true
    end

    """
    Test bound consistency: sum of ab_i should match ab for factorized flows
    """
    function test_bound_consistency(flow::FactorizedDynamics, ξ::SkeletonPoint, c::AbstractVector)

        cache_full = PDMPSamplers.initialize_cache(flow, FullGradient((out,x)->out.=0), ThinningStrategy(GlobalBounds(c[1], length(ξ.x))), 0.0, ξ)
        cache_coord = PDMPSamplers.initialize_cache(flow, CoordinateWiseGradient((x,i)->0.0), ThinningStrategy(LocalBounds(c)), 0.0, ξ)

        # Get global bounds
        a_global, b_global, _ = PDMPSamplers.ab(ξ, c, flow, cache_full)

        # Get sum of coordinate bounds
        a_sum, b_sum = 0.0, 0.0
        for i in eachindex(ξ.x)
            a_i, b_i = PDMPSamplers.ab_i(i, ξ, c, flow, cache_coord)
            a_sum += a_i
            b_sum += b_i
        end

        # They should be approximately equal (up to the safety constants c)
        @test isapprox(a_global, a_sum, rtol=0.1)  # Allow for different safety factors
        @test isapprox(b_global, b_sum, rtol=0.1)

        return true
    end

    # Test parameters
    ds = [2, 5, 10]
    flows_and_types = [
        (ZigZag, FactorizedDynamics),
        (BouncyParticle, NonFactorizedDynamics),
        # (Boomerang, NonFactorizedDynamics)
    ]
    # flow_type, dynamics_type = flows_and_types[1]
    @testset "Flow: $flow_type, d=$d" for (flow_type, dynamics_type) in flows_and_types, d in ds

        # Generate test problem
        D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 2.0)

        # Create flow
        flow = flow_type(inv(cov(D)), mean(D))

        # Test multiple random initial conditions
        for _ in 1:5
            x0 = randn(d)
            θ0 = PDMPSamplers.initialize_velocity(flow, d)
            ξ0 = SkeletonPoint(x0, θ0)

            # Conservative bounds
            c_vec = fill(1e-6 / d, d)

            # Test global bounds
            test_rate_bounds(flow, ξ0, ∇f!, c_vec)

            # Test coordinate bounds for factorized flows
            if dynamics_type == FactorizedDynamics
                @test test_coordinate_bounds(flow, ξ0, ∂fxᵢ, c_vec)
                @test test_bound_consistency(flow, ξ0, c_vec)
            end
        end
    end
end
