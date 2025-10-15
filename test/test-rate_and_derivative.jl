# broken until Mooncake works with Julia 1.12
# @testset "Test get_rate_and_deriv against ad" begin


#     # this one is more useful in general though
#     # TODO: the shenanigans with t as a Ref{Float64} needed to make this work with AD
#     # hint at that the design is suboptimal!
#     function rate_fun(t, x, θ, grad, flow, cache)
#         state = PDMPState(0.0, copy(x), copy(θ))
#         move_forward_time!(state, t, flow)
#         # ξ = SkeletonPoint(copy(x), copy(θ))
#         # ξₜ = move_forward_time(ξ, t, flow)
#         # state = PDMPState(0.0, ξₜ)
#         ∇ϕ = compute_gradient!(state, grad, flow, cache)
#         λ(state.ξ, ∇ϕ, flow)
#     end

#     pdmp_type = ZigZag
#     pdmp_types = (ZigZag, BouncyParticle, Boomerang)

#     # TODO: fix this for the Boomerang!
#     @testset "$pdmp_type" for pdmp_type in pdmp_types

#         d = 5
#         D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 1.0)
#         # flow = Boomerang(inv(cov(D)), mean(D), 0.1)
#         flow = pdmp_type(inv(cov(D)), mean(D)) # TODO: Boomerang does not work in this case?
#         flow = pdmp_type(d)
#         x = randn(d)#mean(D) + .2 .* randn(d)
#         θ = PDMPSamplers.initialize_velocity(flow, d)
#         grad = FullGradient(∇f!)
#         ξ = SkeletonPoint(x, θ)
#         state = PDMPState(0.0, ξ)
#         cache = PDMPSamplers.add_gradient_to_cache(PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(1.0, 5)), 0.0, ξ), ξ)

#         xvals = 0:.01:10

#         f = t->rate_fun(t, x, θ, grad, flow, cache)
#         ad_type = DI.AutoMooncake()
#         # ad_type = DI.AutoForwardDiff()
#         # ad_type = DI.AutoEnzyme()
#         prep = DI.prepare_derivative(f, ad_type, 0.0)

#         n = length(xvals)# - 1
#         yvals1, dvals1 = similar(xvals, n), similar(xvals, n)
#         yvals2, dvals2 = similar(xvals, n), similar(xvals, n)
#         out = similar(x)
#         state_ = copy(state)
#         for i in eachindex(xvals)

#             yvals1[i], dvals1[i] = DI.value_and_derivative(f, prep, ad_type, xvals[i])
#             local state_ = move_forward_time(state, xvals[i], flow)
#             grad_func = PDMPSamplers.make_grad_U_func(state_, flow, grad, cache)
#             yvals2[i], dvals2[i] = PDMPSamplers.get_rate_and_deriv(state_, flow, (grad_func, (x, v)->∇²f!(out, x, v)), false)
#         end
#         @test yvals1 ≈ yvals2
#         @test dvals1 ≈ dvals2
#         # [dvals1 dvals2]

#         # visual inspection
#         # fig = Figure()
#         # ax = Axis(fig[1, 1])
#         # scatter!(xvals, yvals1, color=:blue)
#         # scatter!(xvals, yvals2, color=:orange)
#         # for i in 1:length(xvals)
#         #     # Tangent line: y = yvals1[i] + dvals1[i] * (t - xvals[i])
#         #     # Short segment: from xvals[i] - Δ to xvals[i] + Δ
#         #     Δ = (i == 1 || i == length(xvals)) ? (xvals[2] - xvals[1]) / 2 : (xvals[i+1] - xvals[i-1]) / 4
#         #     t_start = xvals[i] - Δ
#         #     t_end   = xvals[i] + Δ
#         #     t_segment = [t_start, t_end]
#         #     y_segment1 = yvals1[i] .+ dvals1[i] .* (t_segment .- xvals[i])
#         #     y_segment2 = yvals2[i] .+ dvals2[i] .* (t_segment .- xvals[i])
#         #     lines!(ax, t_segment, y_segment1, color=:purple, linewidth=2)
#         #     lines!(ax, t_segment, y_segment2, color=:yellow, linewidth=2)
#         # end
#         # fig
#     end
# end
