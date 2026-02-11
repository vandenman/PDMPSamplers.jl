@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "General properties of PDMP flows" begin

    d = 25
    x = Vector{Float64}(undef, d)
    θ = Vector{Float64}(undef, d)
    ∇ϕx = Vector{Float64}(undef, d)
    θ2 = Vector{Float64}(undef, d)

    for pdmp_type in (Boomerang,)

        D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 2.0)
        flow = pdmp_type(inv(Symmetric(cov(D))), mean(D))

        sqrt_inv_Σ = sqrt(flow.Γ)

        for _ in 1:5
            randn!(x)
            randn!(θ)
            randn!(∇ϕx)
            copyto!(θ2, θ)

            ξ = SkeletonPoint(x, θ2)

            grad = FullGradient((out, x) -> out .= x)
            cache = PDMPSamplers.add_gradient_to_cache(
                PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(1.0, d)), 0.0, ξ),
                ξ
            )

            # Equation 4 of Bierkens et al., (2020), http://proceedings.mlr.press/v119/bierkens20a.html
            rhs = -dot(ξ.θ, ∇ϕx)
            PDMPSamplers.reflect!(ξ, ∇ϕx, flow, cache)
            lhs = dot(ξ.θ, ∇ϕx)
            @test lhs ≈ rhs

            # Equation 5 of Bierkens et al., (2020), http://proceedings.mlr.press/v119/bierkens20a.html
            lhs = norm(sqrt_inv_Σ * ξ.θ)
            rhs = norm(sqrt_inv_Σ * θ)
            @test lhs ≈ rhs
        end
    end
end
