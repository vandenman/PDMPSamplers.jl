
function gen_data(::Type{Distributions.MvNormal}, d, η, μ = rand(Normal(0, 5), d), σ = rand(LogNormal(0, 1), d), R = rand(LKJ(d, η)))
    # μ = rand(Normal(0, 5), d)
    # σ = rand(LogNormal(0, 1), d)
    # R = rand(LKJ(d, η))
    Σ = Symmetric(Diagonal(σ) * R * Diagonal(σ))
    Σ_inv = inv(Σ) # could do this in a safer way
    potential = Σ_inv * μ
    buffer = similar(potential)
    D = MvNormal(μ, Σ)

    ∇f! = (out, x) -> begin
        #out .= -gradlogpdf(D, x)
        mul!(buffer, Σ_inv, x)
        buffer .-= potential
        out .= buffer
    end

    ∇²f! = (out, _, v) -> begin
        mul!(out, Σ_inv, v)
    end

    ∂fxᵢ = (x, i) -> dot(view(Σ_inv, :, i), x) - potential[i]
    return D, ∇f!, ∇²f!, ∂fxᵢ
end
function gen_data(::Type{Distributions.ZeroMeanIsoNormal}, d)
    Σ = I(d)
    D = MvNormal(Σ)
    ∇f! = (out, x) -> copyto!(out, x)
    ∇²f! = (out, _, v) -> copyto!(out, v) # == mul!(out, I, v)
    ∂f∂xᵢ = (x, i) -> x[i]
    return D, ∇f!, ∇²f!, ∂f∂xᵢ
end

function gen_data(::Type{Distributions.MvTDist}, d, η, μ = rand(Normal(0, 5), d), σs = rand(LogNormal(0, 1), d); ν=20.0)
    # 1. Generate random parameters for the distribution, same as MvNormal
    cholR = rand(LKJCholesky(d, η))
    lmul!(Diagonal(σs), cholR.L)
    Σ = PDMats.PDMat(cholR)

    # Σ = Diagonal(σs) * R * Diagonal(σs)
    # Σ = Symmetric(Diagonal(σs) * R * Diagonal(σs))

    # Create the distribution object
    D = MvTDist(ν, μ, Σ)

    # 2. Pre-compute quantities for efficiency
    Σ_inv = Matrix(inv(Σ))
    scalar_coeff = (ν + d) / ν

    # Allocate buffers to avoid memory allocation inside the gradient functions
    # These will be "captured" by the closures below
    x_centered = similar(μ)
    mahal_num = similar(μ)

    # TODO: need some kind of struct for this...
    # would be very cool to just pass a joint (e.g., from DynamicPPL!) and then derive all of these either
    # using user-defined functions or from AD, much like LogDensityProblems.
    # Actually, LogDensityProblems is quite nice but often not enough because it doesn't work when specifying only the gradient.

    # 3. Define the in-place gradient function `ϕ2!`
    ∇f! = (out, x) -> begin
        # Compute the numerator and the Mahalanobis distance squared
        x_centered .= x .- μ
        mul!(mahal_num, Σ_inv, x_centered)  # mahal_num = Σ⁻¹(x-μ)
        mahal_sq = dot(x_centered, mahal_num) # mahal_sq = (x-μ)ᵀΣ⁻¹(x-μ)

        # Compute the final gradient
        denominator = 1 + mahal_sq / ν
        out .= (scalar_coeff / denominator) .* mahal_num
    end

    ∇²f! = (out, x, v) -> begin
        x_centered .= x .- μ
        mul!(mahal_num, Σ_inv, x_centered)  # mahal_num = Σ⁻¹(x-μ)
        q = dot(x_centered, mahal_num)
        c1 = (ν + d) / (ν + q)

        # First term: c1 * Σ⁻¹ v
        mul!(out, Σ_inv, v, c1, zero(eltype(out)))

        # Second term: -2 c1 / (ν+q) * (uᵀΣ⁻¹v) * Σ⁻¹u, using dot(mahal_num, v) = uᵀΣ⁻¹v
        scalar = dot(mahal_num, v)
        axpy!(-2 * c1 / (ν + q) * scalar, mahal_num, out)

        return out
    end

    # Define the i-th component gradient function `ϕ2i!`
    # Note: For the t-distribution, calculating one component requires
    # almost all the work of calculating the full vector.
    ∂fxᵢ = (x, i) -> begin
        # We must recalculate the denominator for the given x
        x_centered .= x .- μ
        mul!(mahal_num, Σ_inv, x_centered)
        mahal_sq = dot(x_centered, mahal_num)

        denominator = 1 + mahal_sq / ν

        # Return just the i-th component
        return (scalar_coeff / denominator) * mahal_num[i]
    end


    return D, ∇f!, ∇²f!, ∂fxᵢ
end

struct GaussianMeanModel{T<:MvNormal}
    X::Matrix{Float64}    # n × p observations
    D::T
    prior_μ::Vector{Float64}    # Prior mean (usually zeros)
end

function GaussianMeanModel(n::Integer, μ, Σ, prior_μ = zero(μ))
    D = MvNormal(μ, Σ)
    X = permutedims(rand(D, n))
    return GaussianMeanModel(X, D, prior_μ)
end

# Analytic posterior (for comparison)
function analytic_posterior(obj::GaussianMeanModel)

    X = obj.X
    n = size(X, 1)

    μ₀ = obj.prior_μ
    Σ₀ = I
    Σ = cov(obj.D)
    x̄ = vec(mean(X, dims = 1))

    Σₙ = inv(inv(Σ₀) + n * inv(Σ))
    μₙ = Σₙ * (inv(Σ₀) * μ₀ + n * inv(Σ) * x̄)

    return MvNormal(μₙ, Σₙ)
end

function gen_data(::Type{GaussianMeanModel}, d, n, μ = zeros(d), Σ = I(d))

    obj = GaussianMeanModel(n, μ, Σ)
    D = analytic_posterior(obj)

    Λ = inv(cov(obj.D)) # could do this in a safer way
    # Λ ≈ I # true for now
    Λ0 = inv(I)
    μ0 = obj.prior_μ
    x̄ = vec(mean(obj.X, dims=1))
    buffer = similar(x̄)
    x̄_sub  = similar(x̄)
    Σ_tot_inv = Λ0 + n .* Λ

    indices = Vector{Int}(undef, 0)
    function resample_indices!(n)
        length(indices) != n && resize!(indices, n)
        sample!(eachindex(indices), indices; replace = false)
    end

    # Full negative gradient: ∇f(x) = Σ0⁻¹(x-μ0) + n * Σ⁻¹(x - x̄)
    ∇f! = (out, x) -> begin
        @. buffer = x - μ0
        mul!(out, Λ0, buffer)                 # prior part
        @. buffer = x - x̄
        mul!(out, Λ, buffer, n, 1.0)              # + n * Λ * (x - x̄)
    end

    # Subsampled negative gradient: replace x̄ with unbiased subsample mean
    ∇f_sub! = (out, x) -> begin

        mean!(x̄_sub', view(obj.X, indices, :))   # subsample mean
        @. buffer = x - μ0
        mul!(out, Λ0, buffer)                 # prior part
        @. buffer = x - x̄_sub
        mul!(out, Λ, buffer, n, 1.0)              # + n * Λ * (x - x̄_sub)
    end
    # this is not faster!
    # function foo(x̄_sub, obj, indices)
    #     fill!(x̄_sub, 0.0)
    #     for i in axes(obj.X, 2)
    #         for j in indices
    #             x̄_sub[i] += obj.X[j, i]
    #         end
    #     end
    #     x̄_sub ./= length(indices)
    #     return x̄_sub
    # end
    # foo(x̄_sub, obj, indices)
    # mean!(x̄_sub', view(obj.X, indices, :))
    # @benchmark mean!($x̄_sub', view($obj.X, $indices, :))
    # @benchmark foo($x̄_sub, $obj, $indices)


    ∇²f! = (out, _, v) -> begin
        mul!(out, Σ_tot_inv, v)
    end

    ∇²f_sub! = (out, _, v) -> begin
        # Same Hessian form (no randomness needed), because Hessian doesn’t depend on subsample
        mul!(out, Σ_tot_inv, v)
    end

    ∂fxᵢ = (x, i) -> dot(view(Σ_inv, :, i), x) - potential[i]

    return D, ∇f!, ∇²f!, ∂fxᵢ, ∇f_sub!, ∇²f_sub!, resample_indices!

end


struct LogisticRegressionModel
    X::Matrix{Float64}
    y::Vector{Int}
    prior_μ::Vector{Float64}
    prior_Σ::Matrix{Float64}
    prior_Σ_inv::Matrix{Float64}
end

function gen_data(::Type{LogisticRegressionModel}, d, n,
                  β_true = randn(d),
                  μ0 = zeros(d),
                  Σ0 = I(d))

    # simulate predictors
    X = randn(n, d-1)
    X .-= mean(X, dims=1) # center predictors
    X = hcat(ones(n), X)  # first column is the intercept
    η = X * β_true
    # p = LogExpFunctions.logistic.(η)
    # y = rand.(Bernoulli.(p))
    y = rand.(BernoulliLogit.(η))

    # prior precision
    Λ0 = inv(Σ0)

    obj = LogisticRegressionModel(X, y, μ0, Σ0, Λ0)

    nobs, d = size(X)
    buffer = zeros(d)
    p = similar(η)

    # resample_indices!(nsub)

        # --- control variate anchor state (closed-over) ---
    β_anchor = copy(μ0)                     # initial anchor (you can change it)
    η_anchor = similar(y, Float64)         # η at anchor (length n)
    p_anchor = similar(y, Float64)         # p = logistic(η_anchor)
    G_anchor = zeros(d)                    # full data gradient data-part at anchor: X'*(p_anchor - y)

    # initialize anchor
    mul!(η_anchor, X, β_anchor)
    @. p_anchor = LogExpFunctions.logistic(η_anchor)
    mul!(G_anchor, X', p_anchor .- y)      # data part only, no prior

    # indices + resample helper
    indices = Vector{Int}(undef, 0)
    resample_indices! = m -> begin
        length(indices) != m && resize!(indices, m)
        sample!(1:nobs, indices; replace = false)
    end

    # setter to (re)compute anchor; call whenever you want to update the anchor
    set_anchor! = β0 -> begin
        β_anchor .= β0
        mul!(η_anchor, X, β_anchor)
        @. p_anchor = LogExpFunctions.logistic(η_anchor)
        mul!(G_anchor, X', p_anchor .- y)
        return nothing
    end

    anchor_info = () -> (β_anchor = copy(β_anchor),
                        p_anchor = copy(p_anchor),
                        G_anchor = copy(G_anchor))

    # --- Full negative log posterior gradient: ∇f(β) = Σ0⁻¹(β-μ0) + X' (σ(Xβ) - y) ---
    ∇f! = (out, β) -> begin
        buffer .= β .- μ0
        mul!(out, Λ0, buffer)  # prior part
        # data part
        mul!(η, X, β)
        for i in eachindex(p)
            p[i] = LogExpFunctions.logistic(η[i]) - y[i]
            # p[i] = 1 / (1 + exp(-η[i])) - y[i]
        end
        mul!(out, X', p, 1.0, 1.0)
    end

    # --- Subsampled gradient with control variate ---
    # estimator: prior + G_anchor + (n/m) * ( X_S'*(p_S(β)-y_S) - X_S'*(p_anchor_S - y_S) )
    ∇f_sub_cv! = (out, β) -> begin
        # prior
        buffer .= β .- μ0
        mul!(out, Λ0, buffer)

        # add full-data anchor data-part
        out .+= G_anchor

        # minibatch difference
        m = length(indices)
        iszero(m) && return out   # nothing else to do

        Xi = @view X[indices, :]
        yi = @view y[indices]
        ηc = view(η, eachindex(indices))   # temporary storage for η_i
        pc = view(p, eachindex(indices))
        # compute η_i(β) for batch
        mul!(ηc, Xi, β)                         # η_i = Xi * β

        pc .= LogExpFunctions.logistic.(ηc)     # p_i(β)

        # p_anchor for the batch (cheap indexed access)
        p0_batch = view(p_anchor, indices)

        # batch gradients: g_batch = Xi'*(p_i - y_i), g0_batch = Xi'*(p0_batch - y_i)
        # so diff = Xi'*( (p_i - y_i) - (p0_batch - y_i) ) = Xi'*(p_i - p0_batch)
        # compute diff = Xi'*(p_i - p0_batch)
        ηc .= pc .- p0_batch                  # reuse ηi storage for pi - p0
        mul!(buffer, Xi', ηc)                  # buffer <- Xi' * (pi - p0_batch)

        # Let's work out the full derivation
        # η_i = Xi * β
        # p_i = σ(η_i) = σ(Xi * β)
        # w_i = p_i .- p0_batch_i = σ(Xi * β) - p0_batch_i
        # buffer_i = Xi' * (w_i) = Xi' * (σ(Xi * β) - p0_batch_i)
        # out_i = scale * buffer_i

        scale = (nobs / m)
        out .+= scale .* buffer
    end

    # --- Full Hessian-vector product: ∇²f(β) v = Σ0⁻¹ v + X' ( w .* (Xv) ), w=p*(1-p) ---
    ∇²f! = (out, β, v) -> begin
        mul!(out, Λ0, v)   # prior part

        mul!(η, X, β)
        p .= LogExpFunctions.logistic.(η)
        p .= p .* (1.0 .- p)   # p now holds weights w

        mul!(η, X, v)          # η <- X * v
        η .*= p                # η <- w .* (X * v)
        mul!(buffer, X', η)
        out .+= buffer
    end

    # --- Subsampled Hessian-vector product (no CV here) ---
    ∇²f_sub! = (out, β, v) -> begin
        mul!(out, Λ0, v)

        m = length(indices)
        iszero(m) && return out

        Xi = @view X[indices, :]
        ηc = view(η, eachindex(indices))
        pc = view(p, eachindex(indices))

        mul!(ηc, Xi, β)
        pc .= LogExpFunctions.logistic.(ηc)
        pc .= pc .* (1.0 .- pc)    # now pi holds w_i
        Xv = ηc # rename for clarity
        mul!(Xv, Xi, v)                # X_i * v for batch
        Xv .*= pc
        mul!(buffer, Xi', Xv)         # buffer <- X_S' * ( w_S .* (X_S v) )
        scale = (nobs / m)
        out .+= scale .* buffer
    end

    # out = similar(β_true)
    # ∇f!(out, β_true)
    # set_anchor!(zero(β_true))   # set anchor at truth for testing
    # resample_indices!(n)
    # ∇f_sub_cv!(out, β_true)
    # resample_indices!(nsub)
    # ∇f_sub_cv!(out, β_true)
    # ∇f_sub_cv_samples = stack(_ -> begin
    #         resample_indices!(nsub)
    #         ∇f_sub_cv!(out, β_true)
    #         copy(out)
    # end, 1:100_000, dims = 1
    # )
    # @btime ∇f!($out, $β_true)
    # @btime ∇f_sub_cv!($out, $β_true)
    # @profview_allocs foreach(_->∇f!(out, β_true), 1:10000)

    # ∇f_sub_cv_exp = mean.(eachcol(∇f_sub_cv_samples))
    # ∇f_sub_cv_var = var.(eachcol(∇f_sub_cv_samples))
    # ∇f_sub_cv_exp .- ∇f!(out, β_true)

    # v = randn(d)
    # ∇²f!(out, β_true, v)
    # resample_indices!(n)
    # ∇²f_sub!(out, β_true, v)
    # resample_indices!(nsub)
    # ∇²f_sub!(out, β_true, v)
    # ∇²f_sub_samples = stack(_ -> begin
    #         resample_indices!(nsub)
    #         ∇²f_sub!(out, β_true, v)
    #         copy(out)
    # end, 1:100_000, dims = 1
    # )
    # @btime ∇²f!($out, $β_true, $v)
    # @btime ∇²f_sub!($out, $β_true, $v)

    # ∇f_sub_cv_exp = mean.(eachcol(∇²f_sub_samples))
    # ∇f_sub_cv_var = var.(eachcol(∇²f_sub_samples))
    # ∇f_sub_cv_exp .- ∇²f!(out, β_true, v)

    return obj, ∇f!, ∇²f!, ∇f_sub_cv!, ∇²f_sub!, resample_indices!, set_anchor!, anchor_info, β_true

#= minibatch approach, variance is too high
    # Negative log posterior gradient: ∇f(β) = Σ0⁻¹(β-μ0) + X' (σ(Xβ) - y)
    ∇f! = (out, β) -> begin
        @. buffer = β - μ0
        mul!(out, Λ0, buffer)  # prior part
        # data part
        mul!(ηc, X, β)
        pc .= LogExpFunctions.logistic.(ηc)
        ηc .= pc .- y
        mul!(out, X', ηc, 1.0, 1.0)
    end

    # Subsampled version: scale contributions
    ∇f_sub! = (out, β) -> begin
        @. buffer = β - μ0
        mul!(out, Λ0, buffer)  # prior part

        Xi = @view X[indices, :]
        yi = @view y[indices]
        ηci = view(ηc, 1:length(indices))
        pci = view(pc, 1:length(indices))
        mul!(ηci, Xi, β)
        pci .= LogExpFunctions.logistic.(ηci)
        ηci .= pci .- yi
        mul!(out, Xi', ηci, 1.0, (nobs / length(indices)))
    end
    out = similar(β_true)
    ∇f!(out, β_true)
    resample_indices!(n)
    ∇f!(out, β_true)
    resample_indices!(nsub)
    ∇f_sub!(out, β_true)
    mean(begin
            resample_indices!(nsub)
            ∇f_sub!(out, β_true)
        end
        for _ in 1:100_000
    )

    # Hessian: ∇²f(β) v = Σ0⁻¹ v + X' W (Xv)  with W=diag(p*(1-p))
    ∇²f! = (out, β, v) -> begin
        mul!(out, Λ0, v)
        mul!(ηc, X, β)
        pc .= LogExpFunctions.logistic.(ηc)
        w = pc .* (1 .- pc)
        mul!(ηc, X, v)         # Xv stored in ηc
        ηc .*= w               # elementwise multiply by w
        mul!(out, X', ηc, 1., 1.)
    end

    # Subsampled Hessian-vector product (unbiased by scaling, no allocations)
    ∇²f_sub! = (out, β, v) -> begin
        mul!(out, Λ0, v)
        Xi = @view X[indices, :]
        ηci = view(ηc, 1:length(indices))
        pci = view(pc, 1:length(indices))
        mul!(ηci, Xi, β)
        pci .= LogExpFunctions.logistic.(ηci)
        w = pci .* (1 .- pci)
        mul!(ηci, Xi, v)
        ηci .*= w
        mul!(out, Xi', ηci, (nobs / length(indices)), 1.0)
    end

    return obj, ∇f!, ∇²f!, ∇f_sub!, ∇²f_sub!, resample_indices!
=#
end


struct SpikeAndSlabDist{D1, D2}
    spike_dist::D1
    slab_dist::D2
end

function marginal_pdfs_at_zero(D::Distributions.AbstractMvNormal)
    μ, Σ = mean(D), cov(D)
    [pdf(Normal(μ[i], sqrt(Σ[i, i])), 0.0) for i in 1:length(D)]
end

function marginal_pdfs_at_zero(D::Distributions.MvTDist)
    ν = D.df
    μ = mean(D)
    Σ = Matrix(D.Σ)  # scale matrix (not covariance)
    [pdf(μ[i] + sqrt(Σ[i, i]) * TDist(ν), 0.0) for i in 1:length(D)]
end

function gen_data(::Type{<:SpikeAndSlabDist{<:Bernoulli, <:Distributions.MvTDist}}, d, η)

    prob = rand(d)
    D1 = product_distribution([Bernoulli(prob[i]) for i in 1:d])
    # Zero mean and unit scales: ensures fast sticky mixing while
    # still exercising the position-dependent MvTDist gradient code.
    μ = zeros(d)
    σs = ones(d)
    D2, ∇f!, ∇²f!, ∂fxᵢ = gen_data(Distributions.MvTDist, d, η, μ, σs)
    D = SpikeAndSlabDist(D1, D2)

    κ = prob ./ (1 .- prob) .* marginal_pdfs_at_zero(D2)

    return D, κ, ∇f!, ∇²f!, ∂fxᵢ
end

function gen_data(::Type{<:SpikeAndSlabDist{<:Bernoulli, T}}, d, args...) where T

    prob = rand(d)
    D1 = product_distribution([Bernoulli(prob[i]) for i in 1:d])
    D2, ∇f!, ∇²f!, ∂fxᵢ = gen_data(T, d, args...)
    D = SpikeAndSlabDist(D1, D2)

    κ = prob ./ (1 .- prob) .* marginal_pdfs_at_zero(D2)

    return D, κ, ∇f!, ∇²f!, ∂fxᵢ
end

function gen_data2(::Type{<:SpikeAndSlabDist{<:Bernoulli, T}}, d, prob, args...) where T

    D1 = product_distribution([Bernoulli(prob[i]) for i in 1:d])
    D2, ∇f!, ∇²f!, ∂fxᵢ = gen_data(T, d, args...)
    D = SpikeAndSlabDist(D1, D2)

    κ = prob ./ (1 .- prob) .* marginal_pdfs_at_zero(D2)

    return D, κ, ∇f!, ∇²f!, ∂fxᵢ
end

function gen_data(::Type{<:SpikeAndSlabDist{<:BetaBernoulli, T}}, d, args...) where T

    a, b = 2. + randexp(), 2 .+ randexp()
    D1 = BetaBernoulli(d, a, b)
    D2, ∇f!, ∇²f!, ∂fxᵢ = gen_data(T, d, args...)
    D = SpikeAndSlabDist(D1, D2)

    mpdfs = marginal_pdfs_at_zero(D2)

    κ = (i, x, γ, args...) -> begin

        # critical! do not look at sum(!iszero, x)
        # because that doesn't mean a particle is frozen.
        # instead, use γ which is state.free
        # note that this is only called whenever we're computing the time to stay frozen, so at that point
        # we have γ[i] == false, hence the assert below
        @assert !γ[i] "κ(...) was called but the coordinate to compute the freezing time for (i=$i) is not frozen!?"

        k_free = sum(γ)
        n_tot  = length(x)

        # should never happen because this should only be called when we're computing the time to stay frozen
        k_free == n_tot && throw(error("All parameters are free, cannot compute κ!"))

        # inclusion odds conditional on current sticky state
        prior_incl_odds = (a + k_free) / (b + n_tot - k_free - 1)

        if n_tot - k_free - 1 + b <= 0
            # or error?
            # @warn "Inclusion odds are infinite"
            return Inf
        end

        return prior_incl_odds * mpdfs[i]
    end

    # a, b = randexp(), randexp()
    # n_tot = n = 5
    # k_free = n # <- fails
    # for k_free in 0:n-1
    #     e1 = ((k_free+1) / (n_tot - k_free)) * pdf(BetaBinomial(n_tot, a, b), k_free + 1) / pdf(BetaBinomial(n_tot, a, b), k_free)
    #     e2 =  (a + k_free) / (n_tot - k_free - 1 + b)
    #     @assert e1 ≈ e2 "k_free = $k_free"
    # end

    return D, κ, ∇f!, ∇²f!, ∂fxᵢ
end


"""
    StickyTime(c, z0, vz)

Distribution of unfreeze times for hierarchical Beta–Bernoulli
(logit–θ parametrization) with slab density c=π_i(0), current logit z0,
and velocity vz of z.

- If vz == 0: this is just Exponential(c * exp(z0)).
- If vz != 0: τ = (1/vz) * log(1 + (vz/c*exp(z0)) * E), E ~ Exp(1).
"""
struct StickyTime <: ContinuousUnivariateDistribution
    c::Float64   # slab density at zero (π_i(0))
    z0::Float64  # current logit θ
    vz::Float64  # current velocity of z
end

function Base.rand(rng::Random.AbstractRNG, d::StickyTime)
    λ0 = d.c * exp(d.z0)
    if iszero(d.vz)
        return rand(rng, Exponential(λ0))
    else
        E = rand(rng, Exponential())
        arg = 1 + (d.vz * E) / λ0
        return arg > 0 ? log(arg) / d.vz : Inf
    end
end


"""
    BetaBernoulliHierarchical(n, a, b)

Hierarchical version of the BetaBernoulli distribution:
γᵢ ~ Bernouilli(p)
p  ~ Beta(a, b)
but _without_ integrating out p (which is what BetaBernoulli does).

primarily used for testing purposes.
"""
# type for testing that refers to
struct BetaBernoulliHierarchical{T<:BetaBernoulli} <: DiscreteMultivariateDistribution
    d::T
end
Base.length(d::BetaBernoulliHierarchical) = length(d.d)
Base.eltype(::Type{<:BetaBernoulliHierarchical}) = eltype(d.d)
Distributions.insupport(d::BetaBernoulliHierarchical, x::AbstractVector{<:Integer}) = Distributions.insupport(d.d, x)
Distributions._rand!(rng::Random.AbstractRNG, d::BetaBernoulliHierarchical, x::AbstractVector) = rand!(rng, d.d, x)
Distributions._logpdf(d::BetaBernoulliHierarchical, x::AbstractVector{<:Integer}) = Distributions._logpdf(d.d, x)
Statistics.mean(d::BetaBernoulliHierarchical) = mean(d.d)



function gen_data(::Type{<:SpikeAndSlabDist{<:BetaBernoulliHierarchical, T}}, d, args...) where T

# hyperprior for θ
    a, b = 2. + randexp(), 2. + randexp()

    # prior over indicators with explicit θ
    D1 = BetaBernoulliHierarchical(BetaBernoulli(d - 1, a, b))   # marker only

    # continuous slab part: first d-1 coordinates
    D2, ∇f_base!, ∇²f_base!, ∂fxᵢ_base = gen_data(T, d - 1, args...)
    D = SpikeAndSlabDist(D1, D2)

    # precompute slab density at zero
    mpdfs = marginal_pdfs_at_zero(D2)

    # κ: conditional inclusion odds × slab density at 0
    κ = (i, x, γ, θ) -> begin
        @assert 1 <= i <= d-1 "κ only defined for slab coordinates"
        # z = x[end]
        # θ = LogExpFunctions.logistic(z)
        # prior_incl_odds = θ / (1 - θ)
        # return prior_incl_odds * marginal_pdfs_at_zero[i]
        z0 = x[end]
        vz = θ[end]
        c  = mpdfs[i]
        return StickyTime(c, z0, vz)
    end

    # gradient of potential
    ∇f! = function (out, x)
        ∇f_base!(view(out, 1:d-1), view(x, 1:d-1))
        z = x[d]
        θ = LogExpFunctions.logistic(z)
        out[d] = -a + (a + b) * θ
        return out
    end

    # Hessian–vector product
    ∇²f! = function (out, x, v)
        ∇²f_base!(view(out, 1:d-1), view(x, 1:d-1), view(v, 1:d-1))
        z = x[d]
        θ = LogExpFunctions.logistic(z)
        out[d] = (a + b) * θ * (1 - θ) * v[d]
        return out
    end

    # single partial derivative
    ∂fxᵢ = function (x, i)
        if i <= d-1
            return ∂fxᵢ_base(view(x, 1:d-1), i)
        elseif i == d
            z = x[d]
            θ = LogExpFunctions.logistic(z)
            return -a + (a + b) * θ
        else
            throw(ArgumentError("i out of bounds"))
        end
    end

    return D, κ, ∇f!, ∇²f!, ∂fxᵢ
end


data_name(::Type{Distributions.ZeroMeanIsoNormal}, d) = "N(0, I($d))"
data_name(::Type{Distributions.MvNormal}, d) = "N(μ, Σ$d)"
data_name(::Type{Distributions.FullNormal}, d) = "N(μ, Σ$d)"
data_name(::Type{<:Distributions.MvTDist}, d) = "T(ν, μ, Σ$d)"
data_name(::Type{<:Distributions.AbstractMvNormal}, d) = "MvNormal($d)"

data_name(::Type{<:Distributions.Product{<:Any, T, <:Any}}, d) where T = data_name(T, d)
data_name(::Type{<:Distributions.Bernoulli}, d) = "Bern"
data_name(::Type{<:BetaBernoulli}, d) = "BetaBern"
data_name(::Type{<:BetaBernoulliHierarchical}, d) = "Bern(p), p ~ Beta"
data_name(::Type{SpikeAndSlabDist{D1, D2}}, d) where {D1, D2} = "δ₀ + (1 - δ₀)$(data_name(D2, d)), π₀ ~ $(data_name(D1, d))"

function _flow_name(trace)
    flow = trace.flow
    if flow isa PDMPSamplers.PreconditionedDynamics
        return "Precond($(nameof(typeof(flow.dynamics))))"
    else
        return string(nameof(typeof(flow)))
    end
end

function _f3(x)
    s = string(round(x; digits=3))
    i = findfirst('.', s)
    i === nothing && (s *= "."; i = length(s))
    return s * "0" ^ max(0, 3 - (length(s) - i))
end

function _isapprox_closeness(a, b; rtol::Real=0.0, atol::Real=0.0)
    err = norm(a .- b)
    tol = max(atol, rtol * max(norm(a), norm(b)))
    return tol > 0 ? err / tol : (iszero(err) ? 0.0 : Inf)
end

function test_approximation(trace::PDMPSamplers.AbstractPDMPTrace, D::Distributions.AbstractMvNormal)

    min_ess = minimum(ess(trace))
    if min_ess < 500
        show_test_diagnostics && @info "Skipping test_approximation (low ESS)" min_ess
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)
    mean_rtol  = 0.2  + 3.0  * mc
    mean_atol  = 0.2  + 3.0  * mc
    cov_rtol   = 0.4  + 10.0 * mc
    quant_rtol = 0.2  + 3.0  * mc

    trace_mean = mean(trace)
    trace_cov = cov(trace)

    @test isapprox(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    cov_rtol < 1.0 && @test isapprox(trace_cov, cov(D); rtol=cov_rtol)

    # test quantiles in the first dimension
    probs = .1:.1:.99
    dist = Normal(mean(D)[1], sqrt(cov(D)[1, 1]))
    q_expected = map(Base.Fix1(quantile, dist), probs)
    q_observed = quantile(trace, collect(probs); coordinate=1)
    quant_rtol < 1.0 && @test isapprox(q_observed, q_expected; rtol=quant_rtol)

    c_mean  = _isapprox_closeness(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    c_cov   = _isapprox_closeness(trace_cov, cov(D); rtol=cov_rtol)
    c_quant = _isapprox_closeness(q_observed, q_expected; rtol=quant_rtol)
    failed = c_mean > 1.0 || (cov_rtol < 1.0 && c_cov > 1.0) || (quant_rtol < 1.0 && c_quant > 1.0)
    if failed || show_test_diagnostics
        d = length(D)
        label = failed ? "FAIL" : "ok  "
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(data_name(typeof(D), d), 16)) | ESS=$(lpad(round(Int, min_ess), 7)) | c_mean=$(_f3(c_mean)) | c_cov=$(_f3(c_cov)) | c_quant=$(_f3(c_quant))")
    end
end

function test_approximation(trace::PDMPSamplers.AbstractPDMPTrace, D::Distributions.MvTDist)

    ν_D, μ_D, Σ_D = Distributions.params(D)

    min_ess = minimum(ess(trace))
    if min_ess < 500
        show_test_diagnostics && @info "Skipping test_approximation (low ESS)" min_ess
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)
    kurtosis_factor = ν_D > 4 ? sqrt((ν_D - 2) / (ν_D - 4)) : 2.0

    mean_rtol  = 0.2  + 3.0  * mc
    mean_atol  = 0.2  + 3.0  * mc
    cov_rtol   = 0.70 + 50.0 * kurtosis_factor * mc
    quant_rtol = 0.30 + 5.0  * kurtosis_factor * mc

    trace_mean = mean(trace)
    trace_cov = cov(trace)

    @test isapprox(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    cov_rtol < 1.0 && @test isapprox(trace_cov, cov(D); rtol=cov_rtol)

    # test quantiles in the first dimension
    qprobs = .1:.1:.99
    marginal_dist = μ_D[1] + sqrt(Σ_D[1, 1]) * Distributions.TDist(ν_D)
    q_expected = map(Base.Fix1(quantile, marginal_dist), qprobs)
    q_observed = quantile(trace, collect(qprobs); coordinate=1)
    quant_rtol < 1.0 && @test isapprox(q_observed, q_expected; rtol=quant_rtol)

    c_mean  = _isapprox_closeness(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    c_cov   = _isapprox_closeness(trace_cov, cov(D); rtol=cov_rtol)
    c_quant = _isapprox_closeness(q_observed, q_expected; rtol=quant_rtol)
    failed = c_mean > 1.0 || (cov_rtol < 1.0 && c_cov > 1.0) || (quant_rtol < 1.0 && c_quant > 1.0)
    if failed || show_test_diagnostics
        d = length(D)
        label = failed ? "FAIL" : "ok  "
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(data_name(typeof(D), d), 16)) | ESS=$(lpad(round(Int, min_ess), 7)) | c_mean=$(_f3(c_mean)) | c_cov=$(_f3(c_cov)) | c_quant=$(_f3(c_quant))")
    end
end

function test_approximation(
        samples,
        spike_dist::Union{
            Distributions.Product{<:Any, <:Bernoulli, <:Any},
            BetaBernoulli
        }
    )

    true_incl_probs = mean(spike_dist)

    # test using conjugate credible interval of Binomial model with Beta(1, 1) prior
    a, b = 1, 1
    cri_Δ = 1e-6#.001, 0.025 # <- would be better, but makes the test more flaky
    cri_bounds = [cri_Δ, 1 - cri_Δ]
    ind = falses(size(samples, 1))
    for i in axes(samples, 2)
        ind .= @view(samples[:, i]) .!= 0
        n_eff = MCMCDiagnosticTools.ess(ind)
        p̂ = mean(ind)
        k_eff = p̂ * n_eff

        # k = count(ind)
        # n = length(ind)
        # post = Beta(a + k, b + (n - k))
        post = Beta(a + k_eff, b + (n_eff - k_eff))
        lo, hi = quantile(post, cri_bounds)
        # increase the interval by 300 %
        delta = (hi - lo) * 3.0
        lo -= delta / 2
        hi += delta / 2
        # @show lo, hi
        @test lo <= true_incl_probs[i] <=  hi
    end

    d = size(samples, 2)

    # for d <= 6 we check all possible models
    enumerate_all = d <= 6
    obs_model_counts = StatsBase.countmap([.!iszero.(row) for row in eachrow(samples)])
    no_samples = size(samples, 1)
    obs_model_probs  = Dict{keytype(obs_model_counts), Float64}(k => v / no_samples for (k, v) in obs_model_counts)
    theoretical_probs = Dict{keytype(obs_model_counts), Float64}()
    if enumerate_all
        modelspace = Iterators.map(BitVector, Iterators.product((0:1 for _ in 1:d)...))
        for model in modelspace
            theoretical_probs[model] = pdf(spike_dist, model)
            get!(obs_model_probs, model, 0.0) # ensure all models are present
        end
    else
        for k in keys(obs_model_counts)
            theoretical_probs[k] = pdf(spike_dist, k)
        end
    end
    # should we test/ assert this?
    # keys(theoretical_probs) == keys(obs_model_probs)
    probs_observed = collect(values(obs_model_probs))
    probs_expected = [theoretical_probs[k] for k in keys(obs_model_probs)]

    obs_model_probs[falses(d)]
    theoretical_probs[falses(d)]
    obs_model_probs[trues(d)]
    theoretical_probs[trues(d)]

    # normalize to handle floating point artifacts
    probs_observed ./= sum(probs_observed)
    probs_expected ./= sum(probs_expected)

    # check if the two columns are close enough
    @test maximum(abs(probs_expected[i] - probs_observed[i]) for i in eachindex(probs_expected)) < 1e-2
    # check if the distribution is close enough
    @test isapprox(probs_expected, probs_observed, atol = 1e-2)
    # fig, ax, _ = scatter(probs_expected, probs_observed)
    # ablines!(ax, 0, 1, color = :grey, linestyle = :dash)
    # fig

end

function test_approximation(samples, spike_dist::BetaBernoulliHierarchical)

    d = size(samples, 2)
    subsamples = view(samples, :, 1:d - 1)

    test_approximation(subsamples, spike_dist.d)

    # test θ
    θ = LogExpFunctions.logistic.(samples[:, end])
    θ_mean, θ_var = mean_and_var(θ)

    n_eff = MCMCDiagnosticTools.ess(θ)

    expected_dist = Beta(spike_dist.d.a, spike_dist.d.b)
    expected_mean = mean(expected_dist)
    expected_var  = var(expected_dist)

    @test isapprox(θ_mean, expected_mean, rtol=0.1, atol=0.1)
    @test isapprox(θ_var, expected_var,   rtol=0.2, atol=0.2)

    qprobs = .01:.01:.99
    k = length(qprobs)

    q_observed = quantile(θ, qprobs)
    q_expected = quantile(expected_dist, qprobs)

    n_eff = MCMCDiagnosticTools.ess(θ)

    if n_eff < 200
        isinteractive() && @info "Skipping θ (too few effective draws)" n_eff
        return nothing
    end

    dens = pdf.(expected_dist, q_expected)

    k = length(qprobs)
    Σ = zeros(k, k)
    for j in 1:k, r in j:k
        p, rprob = qprobs[j], qprobs[r]
        Σ[j,r] = (min(p,rprob) - p*rprob) / (n_eff * dens[j] * dens[r])
        Σ[r,j] = Σ[j,r]
    end

    # regularize
    ϵ = 1e-12 * maximum(view(Σ, diagind(Σ)))
    Σ += ϵ * I

    d = q_observed .- q_expected
    T = dot(d, Σ \ d)

    # compare to χ² quantile
    thresh = quantile(Chisq(k), 0.95)  # or 0.99
    @test T <= thresh


end

"""
    test_approximation(trace, D::SpikeAndSlabDist)

Test a sticky PDMP trace against a spike-and-slab target using analytic
trace estimators instead of discretization.

Uses the identity that for a spike-and-slab model:

    E[x_i]         = p_i * μ_i^slab       (full-model mean)
    mean(trace)[i] ≈ E[x_i]

so the conditional slab mean is recoverable as:

    mean(trace) ./ inclusion_probs(trace) ≈ μ_slab
"""
function test_approximation(trace, D::SpikeAndSlabDist)

    d = length(D.slab_dist)

    min_ess = minimum(ess(trace))
    if min_ess < 500
        show_test_diagnostics && @info "Skipping test_approximation SpikeAndSlab (low ESS)" min_ess
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)

    est_incl_probs = inclusion_probs(trace)
    true_incl_probs = mean(D.spike_dist)

    # Use a norm-based check: the per-coordinate `all(...)` test is too strict for d >= 5 because
    # the probability that at least one coordinate exceeds the tolerance grows with d.
    # A norm-based tolerance scales correctly: atol * sqrt(d).
    incl_atol_per = D.slab_dist isa Distributions.MvTDist ? (0.15 + 1.5 * mc) : (0.06 + 1.5 * mc)
    incl_atol = incl_atol_per * sqrt(d)

    @test isapprox(est_incl_probs, true_incl_probs; atol=incl_atol)

    est_mean = mean(trace)
    mean_rtol = 0.15 + 3.0 * mc
    mean_atol = 0.15 + 3.0 * mc

    true_full_mean = true_incl_probs .* mean(D.slab_dist)
    @test isapprox(est_mean[1:d], true_full_mean; rtol=mean_rtol, atol=mean_atol)

    if all(est_incl_probs .>= 0.1)
        est_slab_mean = est_mean[1:d] ./ est_incl_probs[1:d]
        @test isapprox(est_slab_mean, mean(D.slab_dist); rtol=mean_rtol, atol=mean_atol)
    end

    c_incl = _isapprox_closeness(est_incl_probs, true_incl_probs; atol=incl_atol)
    c_full = _isapprox_closeness(est_mean[1:d], true_full_mean; rtol=mean_rtol, atol=mean_atol)
    failed = c_incl > 1.0 || c_full > 1.0
    if failed || show_test_diagnostics
        label = failed ? "FAIL" : "ok  "
        dname = data_name(typeof(D), d)
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(dname, 40)) | ESS=$(lpad(round(Int, min_ess), 7)) | c_incl=$(_f3(c_incl)) | c_full=$(_f3(c_full))")
    end
end
