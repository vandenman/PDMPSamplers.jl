@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Sticky PDMP Sampler Tests" begin

    pdmp_types = (ZigZag, BouncyParticle, Boomerang, PreconditionedZigZag, PreconditionedBPS)
    factorized_gradient_types = (FullGradient,)
    nonfactorized_gradient_types = (FullGradient,)

    get_gradient_types(::Type{<:FactorizedDynamics}) = factorized_gradient_types
    get_gradient_types(::Type{<:NonFactorizedDynamics}) = nonfactorized_gradient_types
    get_gradient_types(::Type{<:PreconditionedDynamics{<:AbstractPreconditioner,T}}) where T = get_gradient_types(T)

    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient}) = (ThinningStrategy, GridThinningStrategy,)
    get_algorithm_types(::Type{<:PreconditionedDynamics}, ::Type{<:FullGradient}) = (GridThinningStrategy,)
    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:CoordinateWiseGradient}) = (ThinningStrategy,)

    data_types = (
        SpikeAndSlabDist{Bernoulli,ZeroMeanIsoNormal},
        SpikeAndSlabDist{BetaBernoulli,ZeroMeanIsoNormal},
        SpikeAndSlabDist{Bernoulli,Distributions.MvTDist},
    )

    ds = (2, 5,)
    data_args = Dict(
        SpikeAndSlabDist{Bernoulli,ZeroMeanIsoNormal}     => ds,
        SpikeAndSlabDist{BetaBernoulli,ZeroMeanIsoNormal} => ds,
        SpikeAndSlabDist{Bernoulli,Distributions.MvTDist} => Iterators.product(ds, (2.,)),
    )

    @testset "$pdmp_type" for pdmp_type in pdmp_types
        @testset "$gradient_type" for gradient_type in get_gradient_types(pdmp_type)
            @testset "$algorithm" for algorithm in get_algorithm_types(pdmp_type, gradient_type)
                @testset "$(data_name(data_type, data_arg))" for data_type in data_types, data_arg in data_args[data_type]

                    # MvTDist slab: only ZigZag-based samplers converge reliably
                    if data_type === SpikeAndSlabDist{Bernoulli,Distributions.MvTDist}
                        pdmp_type <: Union{BouncyParticle, Boomerang,
                            PreconditionedDynamics{<:Any, BouncyParticle},
                            PreconditionedDynamics{<:Any, Boomerang}} && continue
                        algorithm === ThinningStrategy && continue
                    end

                    # Use a stable seed for target generation so all samplers face the same target
                    Random.seed!(hash((data_type, data_arg)))
                    D, κ, ∇f!, ∇²f!, ∂fxᵢ = gen_data(data_type, data_arg...)
                    # Re-seed for the sampler run (initial conditions, etc.)
                    Random.seed!(hash((pdmp_type, gradient_type, algorithm, data_type, data_arg)))

                    d = first(data_arg)
                    T = data_type === SpikeAndSlabDist{Bernoulli,Distributions.MvTDist} ? 1_000_000.0 : 300_000.0
                    c0 = pdmp_type === ZigZag ? 1e-4 : 1e-2

                    flow = pdmp_type(inv(Symmetric(cov(D.slab_dist))), mean(D.slab_dist))

                    alg0 = if algorithm === ThinningStrategy
                        if gradient_type === CoordinateWiseGradient
                            ThinningStrategy(LocalBounds(fill(c0, d)))
                        else
                            ThinningStrategy(GlobalBounds(c0, d))
                        end
                    else
                        GridThinningStrategy(; N=50)
                    end
                    alg = if κ isa Function
                        can_stick = trues(d)
                        Sticky(alg0, κ, can_stick)
                    else
                        Sticky(alg0, κ)
                    end

                    x0 = randn(d)
                    θ0 = PDMPSamplers.initialize_velocity(flow, d)
                    ξ0 = SkeletonPoint(x0, θ0)

                    grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)
                    model = PDMPModel(d, grad, ∇²f!)
                    trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

                    acceptance_prob = stats.reflections_accepted / stats.reflections_events
                    if !(flow isa Boomerang)
                        @test acceptance_prob > 0.55
                    end

                    test_approximation(trace, D)

                end
            end
        end
    end

end
