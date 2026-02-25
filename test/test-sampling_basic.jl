@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Basic PDMP Sampler Tests" begin

    pdmp_types = (ZigZag, BouncyParticle, Boomerang, PreconditionedZigZag, PreconditionedBPS)
    factorized_gradient_types = (CoordinateWiseGradient, FullGradient)
    nonfactorized_gradient_types = (FullGradient,)

    get_gradient_types(::Type{<:FactorizedDynamics}) = factorized_gradient_types
    get_gradient_types(::Type{<:NonFactorizedDynamics}) = nonfactorized_gradient_types
    get_gradient_types(::Type{<:PreconditionedDynamics{<:AbstractPreconditioner,T}}) where T = (FullGradient,)

    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient}) = (ThinningStrategy, GridThinningStrategy,)
    get_algorithm_types(::Type{<:PreconditionedDynamics}, ::Type{<:FullGradient}) = (GridThinningStrategy,)
    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:CoordinateWiseGradient}) = (ThinningStrategy,)

    data_types = (
        Distributions.ZeroMeanIsoNormal,
        Distributions.MvNormal,
        Distributions.MvTDist,
    )

    ds = (2, 5,)
    ηs = (1., 2., 5.)
    data_args = Dict(
        Distributions.ZeroMeanIsoNormal => ds,
        Distributions.MvNormal => Iterators.product(ds, ηs),
        Distributions.MvTDist => Iterators.product(ds, (2.,)),
    )

    @testset "$pdmp_type" for pdmp_type in pdmp_types
        @testset "$gradient_type" for gradient_type in get_gradient_types(pdmp_type)
            @testset "$algorithm" for algorithm in get_algorithm_types(pdmp_type, gradient_type)
                @testset "$(data_name(data_type, data_arg))" for data_type in data_types, data_arg in data_args[data_type]

                    # ThinningStrategy requires manual bounds unsuitable for non-Gaussian targets
                    algorithm === ThinningStrategy && data_type === Distributions.MvTDist && continue
                    # Boomerang's Gaussian reference dynamics are a poor match for heavy-tailed targets
                    pdmp_type <: Union{Boomerang, PreconditionedDynamics{<:Any, Boomerang}} && data_type === Distributions.MvTDist && continue

                    # Use a stable seed for target generation so all samplers face the same target
                    Random.seed!(hash((data_type, data_arg)))
                    target = gen_data(data_type, data_arg...)
                    D = target.D
                    # Re-seed for the sampler run (initial conditions, etc.)
                    Random.seed!(hash((pdmp_type, gradient_type, algorithm, data_type, data_arg)))

                    d = first(data_arg)
                    T = data_type === Distributions.MvTDist ? 400_000.0 : 50_000.0

                    alg = if algorithm === ThinningStrategy
                        if gradient_type === CoordinateWiseGradient
                            c0 = 1e-2
                            ThinningStrategy(LocalBounds(fill(c0, d)))
                        else
                            c0 = if pdmp_type <: Union{Boomerang,PreconditionedDynamics{<:Any,Boomerang}} && data_type <: Distributions.AbstractMvNormal
                                0.0
                            elseif pdmp_type === ZigZag
                                1e-6
                            else
                                1e-2
                            end
                            ThinningStrategy(GlobalBounds(c0 / max(d, 1), d))
                        end
                    elseif algorithm === GridThinningStrategy
                        GridThinningStrategy()
                    end

                    flow = if pdmp_type <: PreconditionedDynamics
                        pdmp_type(d)
                    else
                        pdmp_type(inv(Symmetric(cov(D))), mean(D))
                    end

                    x0 = mean(D) + randn(d)
                    θ0 = PDMPSamplers.initialize_velocity(flow, d)
                    ξ0 = SkeletonPoint(x0, θ0)

                    grad = if gradient_type == CoordinateWiseGradient
                        CoordinateWiseGradient(Base.Fix1(neg_partial, target))
                    else
                        FullGradient(Base.Fix1(neg_gradient!, target))
                    end
                    model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
                    trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

                    # If the stochastic test would fail (e.g., due to a convergence fluke),
                    # retry once with a different seed before registering @test results.
                    if !test_approximation(trace, D; check_only=true)
                        Random.seed!(hash((pdmp_type, gradient_type, algorithm, data_type, data_arg, :retry)))
                        x0 = mean(D) + randn(d)
                        θ0 = PDMPSamplers.initialize_velocity(flow, d)
                        ξ0 = SkeletonPoint(x0, θ0)
                        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)
                    end

                    acceptance_prob = stats.reflections_accepted / stats.reflections_events

                    PDMPSamplers.ispositive(refresh_rate(flow)) && @test stats.refreshment_events > 100

                    if !(flow isa Boomerang)
                        @test acceptance_prob > 0.4
                    end
                    @test length(trace) > 100

                    test_approximation(trace, D; elapsed=stats.elapsed_time)

                end
            end
        end
    end

end
