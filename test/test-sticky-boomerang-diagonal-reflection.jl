using LinearAlgebra

function _sticky_boomerang_fixture(d, free_fraction; seed=9182)
    rng = Xoshiro(seed)
    covariance = exp.(range(log(0.03), log(17.0); length=d))
    precision = 1.0 ./ covariance
    diagonal_flow = PDMPSamplers.MutableBoomerang(
        Diagonal(precision), zeros(d), 0.1)
    dense_flow = PDMPSamplers.MutableBoomerang(
        Matrix(Diagonal(precision)), zeros(d), 0.1)
    nfree = max(1, round(Int, d * free_fraction))
    free = falses(d)
    free[randperm(rng, d)[1:nfree]] .= true
    theta = randn(rng, d)
    gradient = randn(rng, d)
    state = StickyPDMPState(0.0,
        SkeletonPoint(randn(rng, d), copy(theta)), copy(free))
    cache = (; z=zeros(d), tmp=zeros(d))
    return (; covariance, diagonal_flow, dense_flow, free, theta,
        gradient, state, cache)
end

@inline function _reflect_diagonal_fixture!(state, gradient, flow, cache)
    PDMPSamplers.reflect!(state, gradient, flow, cache)
end

@testset "diagonal sticky Boomerang reflection" begin
    for d in (75, 525), fraction in (0.10, 0.25, 0.50, 1.0)
        fixture = _sticky_boomerang_fixture(d, fraction)
        diagonal_state = deepcopy(fixture.state)
        dense_state = deepcopy(fixture.state)
        diagonal_cache = (; z=zeros(d), tmp=zeros(d))
        dense_cache = (; z=zeros(d), tmp=zeros(d))
        frozen_before = copy(fixture.theta[.!fixture.free])

        PDMPSamplers.reflect!(diagonal_state, fixture.gradient,
            fixture.diagonal_flow, diagonal_cache)
        PDMPSamplers._reflect_sticky_boomerang_generic!(dense_state,
            fixture.gradient, fixture.dense_flow, dense_cache)

        @test diagonal_state.ξ.θ[fixture.free] ≈
            dense_state.ξ.θ[fixture.free] rtol=2e-13 atol=2e-13
        @test diagonal_state.ξ.θ[.!fixture.free] == frozen_before
        @test dense_state.ξ.θ[.!fixture.free] == frozen_before

        before_derivative = dot(fixture.theta[fixture.free],
            fixture.gradient[fixture.free])
        after_derivative = dot(diagonal_state.ξ.θ[fixture.free],
            fixture.gradient[fixture.free])
        @test after_derivative ≈ -before_derivative rtol=2e-12 atol=2e-12

        before_energy = sum(abs2(fixture.theta[i]) /
            fixture.covariance[i] for i in eachindex(fixture.theta) if fixture.free[i])
        after_energy = sum(abs2(diagonal_state.ξ.θ[i]) /
            fixture.covariance[i] for i in eachindex(fixture.theta) if fixture.free[i])
        @test after_energy ≈ before_energy rtol=5e-12 atol=5e-12

        PDMPSamplers.reflect!(diagonal_state, fixture.gradient,
            fixture.diagonal_flow, diagonal_cache)
        @test diagonal_state.ξ.θ ≈ fixture.theta rtol=5e-12 atol=5e-12

        z = fixture.covariance .* fixture.gradient .* fixture.free
        expected_denominator = dot(fixture.gradient, z)
        expected_coefficient = 2 * dot(fixture.theta[fixture.free],
            fixture.gradient[fixture.free]) / expected_denominator
        dense_denominator = dot(fixture.gradient[fixture.free],
            dense_cache.z[eachindex(findall(fixture.free))])
        @test dense_denominator ≈ expected_denominator rtol=2e-13 atol=2e-13
        @test expected_coefficient ≈ 2 * dot(fixture.theta[fixture.free],
            fixture.gradient[fixture.free]) / dense_denominator
    end

    for free in (trues(12), BitVector([true; falses(11)]),
            BitVector([isodd(i) for i in 1:12]))
        flow = PDMPSamplers.MutableBoomerang(
            Diagonal(exp.(range(-4.0, 4.0; length=12))), zeros(12), 0.1)
        theta = collect(range(-0.8, 1.1; length=12))
        gradient = collect(range(0.4, 2.0; length=12))
        gradient[.!free] .= 91.0
        state = StickyPDMPState(0.0,
            SkeletonPoint(zeros(12), copy(theta)), copy(free))
        cache = (; z=zeros(12), tmp=zeros(12))
        PDMPSamplers.reflect!(state, gradient, flow, cache)
        @test state.ξ.θ[.!free] == theta[.!free]
    end

    zero_flow = PDMPSamplers.MutableBoomerang(Diagonal(ones(9)), zeros(9), 0.1)
    zero_state = StickyPDMPState(0.0,
        SkeletonPoint(zeros(9), collect(1.0:9.0)), trues(9))
    zero_gradient = zeros(9)
    zero_cache = (; z=zeros(9), tmp=zeros(9))
    before = copy(zero_state.ξ.θ)
    @test PDMPSamplers.reflect!(zero_state, zero_gradient, zero_flow,
        zero_cache) === nothing
    @test zero_state.ξ.θ == before

    allocation_fixture = _sticky_boomerang_fixture(525, 0.50)
    allocation_state = deepcopy(allocation_fixture.state)
    allocation_cache = (; z=zeros(525), tmp=zeros(525))
    _reflect_diagonal_fixture!(allocation_state, allocation_fixture.gradient,
        allocation_fixture.diagonal_flow, allocation_cache)
    GC.gc()
    @test @allocated(_reflect_diagonal_fixture!(allocation_state,
        allocation_fixture.gradient, allocation_fixture.diagonal_flow,
        allocation_cache)) == 0
end
