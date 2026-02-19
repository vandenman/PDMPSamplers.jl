using Test
import Distributions
import Random
import SpecialFunctions
import StatsBase

@testset "BetaBernoulli Distribution Tests" begin

    @testset "Constructor Tests" begin

        # (args..., expected_type)
        cases = (
            # Valid constructors
            ((5, 2.0, 3.0),     BetaBernoulli{Float64}),
            ((10, 1, 2),        BetaBernoulli{Int}),
            ((3, 1.5f0, 2.5f0), BetaBernoulli{Float32}),
            # Type promotion
            ((5, 1, 2.0),       BetaBernoulli{Float64}),
            ((5, 1.0f0, 2),     BetaBernoulli{Float32}),
        )

        for (args, T) in cases
            ss = @inferred T BetaBernoulli(args...)
            @test ss isa T
        end

        # Parameter access
        d = BetaBernoulli(8, 2.5, 1.5)
        @test d.n == 8
        @test d.a == 2.5
        @test d.b == 1.5

        # Invalid constructors should throw
        @test_throws ArgumentError BetaBernoulli(0, 1.0, 2.0)   # n <= 0
        @test_throws ArgumentError BetaBernoulli(-1, 1.0, 2.0)  # n < 0
        @test_throws ArgumentError BetaBernoulli(5, 0.0, 2.0)   # a <= 0
        @test_throws ArgumentError BetaBernoulli(5, -1.0, 2.0)  # a < 0
        @test_throws ArgumentError BetaBernoulli(5, 2.0, 0.0)   # b <= 0
        @test_throws ArgumentError BetaBernoulli(5, 2.0, -1.0)  # b < 0
    end

    @testset "Basic Properties" begin
        d1 = BetaBernoulli(5, 2.0, 3.0)
        d2 = BetaBernoulli(10, 1.0, 1.0)

        # Test required interface methods
        @test length(d1) == 5
        @test length(d2) == 10
        @test eltype(BetaBernoulli) == Int
        @test eltype(d1) == Int

        # Type inference
        @inferred length(d1)
        @inferred eltype(BetaBernoulli)
    end

    @testset "Random Sampling Tests" begin
        Random.seed!(123)
        d = BetaBernoulli(8, 2.0, 3.0)

        # Test basic sampling
        x = rand(d)
        @test x isa Vector{Int}
        @test length(x) == length(d) == 8
        @test all(in((0, 1)), x)

        # Test sampling with pre-allocated vector
        x_preallocated = Vector{Int}(undef, 8)
        fill!(x_preallocated, 2) # outside of domain
        @test Random.rand!(d, x_preallocated) === x_preallocated # return value is the original vector
        @test all(in((0, 1)), x_preallocated)

        # Test multiple samples
        samples = rand(d, 100)
        @test size(samples) == (length(d), 100)
        @test all(in((0, 1)), samples)

        # Test that we get different numbers of ones (not always the same)
        num_ones = vec(sum(samples, dims = 2))
        @test length(unique(num_ones)) > 1  # Should have variation

        # Type inference for sampling
        @inferred Vector{Int} rand(d)
        @inferred Random.rand!(d, Vector{Int}(undef, 8))
    end

    @testset "mean Tests" begin
        # Test mean calculation
        n = 10_000
        x = Matrix{Int}(undef, 10, n)
        x1 = view(x, 1:5, :)
        x2 = x
        x3 = view(x, 1:3, :)

        d1 = BetaBernoulli(5, 2.0, 3.0)
        expected_p1 = 2.0 / (2.0 + 3.0)  # = 0.4
        @test StatsBase.mean(d1) ≈ fill(expected_p1, 5)
        Random.rand!(d1, x1)
        @test StatsBase.mean(d1) ≈ StatsBase.mean(x1, dims=2) atol = .05

        d2 = BetaBernoulli(10, 1.0, 1.0)  # Uniform case
        @test StatsBase.mean(d2) ≈ fill(0.5, 10)
        Random.rand!(d2, x2)
        @test StatsBase.mean(d2) ≈ StatsBase.mean(x2, dims=2) atol = .05

        d3 = BetaBernoulli(3, 3.0, 1.0)   # Biased toward 1
        expected_p3 = 3.0 / (3.0 + 1.0)  # = 0.75
        @test StatsBase.mean(d3) ≈ fill(expected_p3, 3)
        Random.rand!(d3, x3)
        @test StatsBase.mean(d3) ≈ StatsBase.mean(x3, dims=2) atol = .05

        # Type inference
        @inferred StatsBase.mean(d1)

    end

    @testset "Variance Tests" begin
        # Test variance calculation
        d1 = BetaBernoulli(5, 2.0, 3.0)
        p1 = 2.0 / (2.0 + 3.0)
        expected_var1 = p1 * (1 - p1)
        @test StatsBase.var(d1) ≈ fill(expected_var1, 5)

        d2 = BetaBernoulli(10, 1.0, 1.0)
        @test StatsBase.StatsBase.var(d2) ≈ fill(0.25, 10)  # p = 0.5, so var = 0.25

        # Type inference
        @inferred StatsBase.StatsBase.var(d1)

        # Test that empirical variance approximates theoretical variance
        Random.seed!(42)
        d = BetaBernoulli(15, 3.0, 2.0)
        samples = [rand(d) for _ in 1:10000]
        sample_matrix = hcat(samples...)  # 15 × 10000
        empirical_var = [StatsBase.StatsBase.var(sample_matrix[i, :]) for i in 1:15]
        theoretical_var = StatsBase.StatsBase.var(d)
        @test empirical_var ≈ theoretical_var atol=0.02
    end

    @testset "Log PDF Tests" begin
        d = BetaBernoulli(4, 2.0, 3.0)

        # Valid binary vectors
        @test Distributions.logpdf(d, [1, 1, 0, 0]) isa Float64
        @test Distributions.logpdf(d, [0, 0, 0, 0]) isa Float64
        @test Distributions.logpdf(d, [1, 1, 1, 1]) isa Float64

        # Test specific cases
        all_zeros = [0, 0, 0, 0]
        all_ones = [1, 1, 1, 1]
        mixed = [1, 0, 1, 0]

        @test isfinite(Distributions.logpdf(d, all_zeros))
        @test isfinite(Distributions.logpdf(d, all_ones))
        @test isfinite(Distributions.logpdf(d, mixed))

        # All configurations with same number of ones should have same probability
        config1 = [1, 1, 0, 0]
        config2 = [1, 0, 1, 0]
        config3 = [0, 1, 1, 0]
        @test Distributions.logpdf(d, config1) ≈ Distributions.logpdf(d, config2)
        @test Distributions.logpdf(d, config2) ≈ Distributions.logpdf(d, config3)

        # Invalid inputs should return -Inf
        @test Distributions.logpdf(d, [1, 2, 0, 0]) == -Inf  # Contains 2
        @test Distributions.logpdf(d, [1, -1, 0, 0]) == -Inf  # Contains -1

        # Wrong dimension should throw
        @test_throws DimensionMismatch Distributions.logpdf(d, [1, 0, 0])      # Too short
        @test_throws DimensionMismatch Distributions.logpdf(d, [1, 0, 0, 0, 1]) # Too long

        # Type inference
        @inferred Distributions.logpdf(d, [1, 0, 1, 0])

        # Test that probabilities sum to 1 (approximately)
        d_small = BetaBernoulli(3, 1.0, 1.0)
        all_configs = [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ]
        total_prob = sum(exp(Distributions.logpdf(d_small, config)) for config in all_configs)
        @test total_prob ≈ 1.0 atol=1e-12
    end

    @testset "Consistency Tests" begin
        Random.seed!(456)
        d = BetaBernoulli(12, 1.5, 2.5)

        # Test that the distribution of number of ones matches BetaBinomial
        n_samples = 10000
        samples = [rand(d) for _ in 1:n_samples]
        num_ones_counts = [sum(s) for s in samples]

        # Compare with BetaBinomial
        bb = Distributions.BetaBinomial(12, 1.5, 2.5)

        # Test that the mean number of ones matches
        @test StatsBase.mean(num_ones_counts) ≈ StatsBase.mean(bb) atol=0.1

        # Test that the variance of number of ones matches
        @test StatsBase.StatsBase.var(num_ones_counts) ≈ StatsBase.var(bb) atol=0.5
    end

    @testset "Edge Cases" begin
        # Very skewed distributions
        d_skewed_low = BetaBernoulli(5, 0.1, 5.0)   # Very low probability
        d_skewed_high = BetaBernoulli(5, 5.0, 0.1)  # Very high probability

        @test StatsBase.mean(d_skewed_low)[1] < 0.1
        @test StatsBase.mean(d_skewed_high)[1] > 0.9

        # Single element
        d_single = BetaBernoulli(1, 2.0, 3.0)
        x_single = rand(d_single)
        @test length(x_single) == 1
        @test x_single[1] ∈ (0, 1)

        # Large n
        d_large = BetaBernoulli(1000, 2.0, 2.0)
        @test length(d_large) == 1000
        @test length(StatsBase.mean(d_large)) == 1000
    end
end