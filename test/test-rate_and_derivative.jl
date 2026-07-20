@isdefined(PDMPSamplers) || begin
    include(joinpath(@__DIR__, "testsetup.jl"))
    import DifferentiationInterface as DI
end

@testset "rate_and_derivative provider overloads" begin
    A = Diagonal([2.0, 3.0])
    grad = x -> A * x
    hvp = (x, v) -> A * v
    vhv = (x, v, w) -> dot(w, A * v)
    x = [1.0, -2.0]
    θ = [0.5, -1.5]
    state = PDMPState(0.0, SkeletonPoint(copy(x), copy(θ)))
    flow = BouncyParticle(2, 0.0)
    cached = grad(x)

    expected_rate = dot(cached, θ)
    expected_deriv = dot(θ, A * θ)

    @test PDMPSamplers.rate_and_derivative(state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cached) ==
        (expected_rate, expected_deriv)

    vhv_provider = PDMPSamplers.VHVProvider(grad, vhv)
    @test PDMPSamplers.rate_and_derivative(state, flow, vhv_provider) ==
        (expected_rate, expected_deriv)
    @test PDMPSamplers.rate_and_derivative(state, flow, vhv_provider, cached) ==
        (expected_rate, expected_deriv)

    stats = PDMPSamplers.StatisticCounter()
    joint = PDMPSamplers.WithStatsJoint((x, v) -> (dot(grad(x), v), dot(v, hvp(x, v))), stats)
    @test PDMPSamplers.rate_and_derivative(state, flow, joint) ==
        (expected_rate, expected_deriv)
    @test stats.∇²f_calls == 1

    @test PDMPSamplers.rate_and_derivative(state, flow, PDMPSamplers.GradientOnlyProvider(grad)) ==
        (expected_rate, 0.0)
    @test PDMPSamplers.rate_and_derivative(state, flow, PDMPSamplers.GradientOnlyProvider(grad), cached) ==
        (expected_rate, 0.0)

    fd = PDMPSamplers.FiniteDiffVHV(grad, zeros(2))
    fd_rate, fd_deriv = PDMPSamplers.rate_and_derivative(state, flow, fd, cached)
    @test fd_rate ≈ expected_rate
    @test fd_deriv ≈ expected_deriv rtol=1e-6
    @test fd.grad_buf == cached

    preconditioned = PreconditionedDynamics(DiagonalPreconditioner([2.0, 4.0]), flow)
    @test PDMPSamplers.rate_and_derivative(state, preconditioned, PDMPSamplers.GradHVPProvider(grad, hvp), cached) ==
        (expected_rate, expected_deriv)
end

@testset "Boomerang cached derivatives and reference multiplication" begin
    A = Diagonal([2.0, 3.0])
    grad = x -> A * x
    hvp = (x, v) -> A * v
    x = [0.25, -0.5]
    θ = [0.75, -0.25]
    state = PDMPState(0.0, SkeletonPoint(copy(x), copy(θ)))
    flow = Boomerang(Diagonal([1.5, 2.5]), zeros(2), 0.0)
    cached = grad(x)

    @test PDMPSamplers.rate_and_derivative(state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cached) ==
        PDMPSamplers.rate_and_derivative(state, flow, PDMPSamplers.GradHVPProvider(grad, hvp))

    out = zeros(2)
    @test PDMPSamplers._reference_mul!(out, flow, x) === out
    @test out ≈ flow.Γ * x

    lowrank = PDMPSamplers.LowRankPrecision(2, 1)
    lowrank.V[:, 1] .= [0.25, -0.5]
    lowrank.Λ[1] = 0.75
    PDMPSamplers.lowrank_precompute!(lowrank)
    lowrank_flow = PDMPSamplers.MutableBoomerang(lowrank, zeros(2), 0.0, 0.0, nothing, nothing, nothing)
    fill!(out, NaN)
    @test PDMPSamplers._reference_mul!(out, lowrank_flow, x) === out
    expected = similar(x)
    PDMPSamplers.lowrank_mul!(expected, lowrank, x, 1.0, 0.0)
    @test out ≈ expected
end

@testset "rate derivative grid capability predicates" begin
    state = PDMPState(0.0, SkeletonPoint([1.0, 2.0], [1.0, -1.0]))
    grad = x -> copy(x)
    hvp = (x, v) -> copy(v)
    provider = PDMPSamplers.GradHVPProvider(grad, hvp)
    no_hvp = PDMPSamplers.GradientOnlyProvider(grad)

    @test PDMPSamplers._rate_aggregation(BouncyParticle(2)) === :scalar
    @test PDMPSamplers._rate_aggregation(Boomerang(2)) === :scalar
    @test PDMPSamplers._rate_aggregation(ZigZag(2)) === :componentwise
    @test PDMPSamplers._rate_aggregation(PreconditionedDynamics(DiagonalPreconditioner([1.0, 2.0]), BouncyParticle(2))) === :scalar
    @test PDMPSamplers._rate_aggregation(PreconditionedDynamics(DiagonalPreconditioner([1.0, 2.0]), ZigZag(2))) === :componentwise

    @test PDMPSamplers._rate_channel_count(state, BouncyParticle(2)) == 1
    @test PDMPSamplers._rate_channel_count(state, ZigZag(2)) == 2
    @test PDMPSamplers._provider_has_directional_derivative(provider)
    @test !PDMPSamplers._provider_has_directional_derivative(no_hvp)
    @test PDMPSamplers._supports_rate_derivatives(provider, BouncyParticle(2))
    @test !PDMPSamplers._supports_rate_derivatives(no_hvp, Boomerang(2))
    @test PDMPSamplers._uses_builtin_grid_provider(provider)
    @test PDMPSamplers._joint_compatible(BouncyParticle(2))
    @test !PDMPSamplers._joint_compatible(ZigZag(2))
    @test PDMPSamplers.max_grid_horizon(BouncyParticle(2)) == 1e10
    @test PDMPSamplers.max_grid_horizon(Boomerang(2)) == 8π

    values = zeros(1, 3)
    derivatives = zeros(1, 3)
    t_grid = [0.0, 0.5, 1.0]
    pd = PreconditionedDynamics(DiagonalPreconditioner([1.0, 1.0]), BouncyParticle(2, 0.0))
    PDMPSamplers.rate_derivatives_for_grid!(values, derivatives, provider, state, pd, t_grid, 3)
    @test values[1, :] ≈ [dot(state.ξ.x .+ t .* state.ξ.θ, state.ξ.θ) for t in t_grid]
    @test derivatives[1, :] ≈ fill(dot(state.ξ.θ, state.ξ.θ), 3)

    @test_throws ArgumentError PDMPSamplers.rate_derivatives_for_grid!(
        values, derivatives, provider, state, ZigZag(2), t_grid, 3)
end
import ForwardDiff

@testset "Test get_rate_and_deriv against ad" begin


    # this one is more useful in general though
    # TODO: the shenanigans with t as a Ref{Float64} needed to make this work with AD
    # hint at that the design is suboptimal!
    function rate_fun(t, x, θ, grad, flow, cache)
        state = PDMPState(zero(t), SkeletonPoint(Vector{Real}(copy(x)), Vector{Real}(copy(θ))))
        move_forward_time!(state, t, flow)
        # ξ = SkeletonPoint(copy(x), copy(θ))
        # ξₜ = move_forward_time(ξ, t, flow)
        # state = PDMPState(0.0, ξₜ)
        ∇ϕ = compute_gradient!(state, grad, flow, cache)
        λ(state.ξ, ∇ϕ, flow)
    end

    # pdmp_type = ZigZag
    pdmp_types = (ZigZag, BouncyParticle, Boomerang)

    # TODO: fix this for the Boomerang!
    @testset "$pdmp_type" for pdmp_type in pdmp_types

        d = 5
        target = gen_data(MvNormal, d, 1.0)
        D = target.D

        Σ = D.Σ
        μ = D.μ
        Σ_inv = inv(Σ) # could do this in a safer way
        potential = Σ_inv * μ
        buffer_real = Vector{Real}(similar(potential))

        ∇f!_real = (out, x) -> begin
            #out .= -gradlogpdf(D, x)
            mul!(buffer_real, Σ_inv, x)
            buffer_real .-= potential
            out .= buffer_real
        end

        # flow = Boomerang(inv(cov(D)), mean(D), 0.1)
        flow = pdmp_type(inv(cov(D)), mean(D)) # TODO: Boomerang does not work in this case?
        flow = pdmp_type(d)
        x = randn(d)#mean(D) + .2 .* randn(d)
        θ = PDMPSamplers.initialize_velocity(flow, d)
        grad = FullGradient(∇f!_real)
        ξ = SkeletonPoint(x, θ)
        state = PDMPState(0.0, ξ)
        cache0 = PDMPSamplers.add_gradient_to_cache(PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(1.0, 5)), 0.0, ξ), ξ)
        # Widen cache buffers to Vector{Real} so ForwardDiff Dual numbers can be stored
        cache_updates = (; ∇ϕx = Vector{Real}(cache0.∇ϕx))
        if haskey(cache0, :z)
            cache_updates = merge(cache_updates, (; z = Vector{Real}(cache0.z)))
        end
        cache = merge(cache0, cache_updates)

        # xvals = sort!(rand(0:.01:10, 25))
        xvals = 0:.01:10

        f = t->rate_fun(t, x, θ, grad, flow, cache)
        ad_type = DI.AutoForwardDiff()
        prep = DI.prepare_derivative(f, ad_type, 0.0)

        n = length(xvals)# - 1
        yvals1, dvals1 = similar(xvals, n), similar(xvals, n)
        yvals2, dvals2 = similar(xvals, n), similar(xvals, n)
        out = similar(x)
        state_ = copy(state)
        for i in eachindex(xvals)

            yvals1[i], dvals1[i] = DI.value_and_derivative(f, prep, ad_type, xvals[i])
            local state_ = move_forward_time(state, xvals[i], flow)
            grad_func = PDMPSamplers.make_grad_U_func(state_, flow, grad, cache)
            provider = PDMPSamplers.GradHVPProvider(
                grad_func, (x, v) -> neg_hvp!(target, out, x, v))
            yvals2[i], dvals2[i] = PDMPSamplers.get_rate_and_deriv(state_, flow, provider, false)
        end
        @test yvals1 ≈ yvals2
        @test dvals1 ≈ dvals2
        # [dvals1 dvals2]

        # visual inspection
        # fig = Figure()
        # ax = Axis(fig[1, 1])
        # scatter!(xvals, yvals1, color=:blue)
        # scatter!(xvals, yvals2, color=:orange)
        # for i in 1:length(xvals)
        #     # Tangent line: y = yvals1[i] + dvals1[i] * (t - xvals[i])
        #     # Short segment: from xvals[i] - Δ to xvals[i] + Δ
        #     Δ = (i == 1 || i == length(xvals)) ? (xvals[2] - xvals[1]) / 2 : (xvals[i+1] - xvals[i-1]) / 4
        #     t_start = xvals[i] - Δ
        #     t_end   = xvals[i] + Δ
        #     t_segment = [t_start, t_end]
        #     y_segment1 = yvals1[i] .+ dvals1[i] .* (t_segment .- xvals[i])
        #     y_segment2 = yvals2[i] .+ dvals2[i] .* (t_segment .- xvals[i])
        #     lines!(ax, t_segment, y_segment1, color=:purple, linewidth=2)
        #     lines!(ax, t_segment, y_segment2, color=:yellow, linewidth=2)
        # end
        # fig
    end
end
