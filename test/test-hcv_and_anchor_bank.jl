@testset "HCVState" begin
    d = 5

    @testset "construction" begin
        hcv = HCVState(d)
        @test size(hcv.C_matrix) == (d, d)
        @test length(hcv.Cv_buf) == d
        @test length(hcv.v_buf) == d
        @test hcv.vstar2 == 1.0 * d  # default σ_star=1.0
        @test !hcv.enabled

        hcv2 = HCVState(d; σ_star=2.0)
        @test hcv2.vstar2 == 4.0 * d
    end

    @testset "update_hcv!" begin
        hcv = HCVState(d)
        s = 10.0  # N/m = 500/50

        # Create synthetic Hessians (diagonal for simplicity)
        H_full_flat = zeros(d * d)
        H_prior_flat = zeros(d * d)
        for j in 1:d
            H_full_flat[(j - 1) * d + j] = -Float64(j)      # H_full[j,j] = -j
            H_prior_flat[(j - 1) * d + j] = -0.01            # H_prior[j,j] = -0.01
        end

        update_hcv!(hcv, H_full_flat, H_prior_flat, s, d)

        @test hcv.enabled
        # C = (1-s)·H_prior - H_full
        # C[j,j] = (1-10)·(-0.01) - (-j) = 0.09 + j
        for j in 1:d
            @test hcv.C_matrix[j, j] ≈ 0.09 + j
        end

        # vstar2 = sum(1/|H_full[j,j]|) = sum(1/j)
        expected_vstar2 = sum(1.0 / j for j in 1:d)
        @test hcv.vstar2 ≈ expected_vstar2
    end

    @testset "apply_hcv_correction!" begin
        hcv = HCVState(3)
        hcv.C_matrix .= [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0]
        hcv.vstar2 = 10.0
        hcv.enabled = true

        out = [1.0, 2.0, 3.0]
        x = [1.5, 2.5, 3.5]
        anchor = [1.0, 2.0, 3.0]
        s = 10.0
        hvp_buf = zeros(3)

        # v = x - anchor = [0.5, 0.5, 0.5]
        # ||v||² = 0.75
        # α = 10 / (10 + 0.75) = 10/10.75
        # C·v = [0.5, 1.0, 1.5]
        # hvp_at_anchor returns mock values
        hvp_mock! = (out, v) -> (out .= v .* 0.1; out)

        out_copy = copy(out)
        apply_hcv_correction!(out, hcv, x, anchor, hvp_mock!, s, hvp_buf)

        v = [0.5, 0.5, 0.5]
        α = 10.0 / (10.0 + 0.75)
        Cv = [0.5, 1.0, 1.5]
        hvp_v = v .* 0.1
        expected = out_copy .+ α .* (s .* hvp_v .+ Cv)

        @test out ≈ expected

        # With disabled HCV, no change
        hcv2 = HCVState(3)
        hcv2.enabled = false
        out2 = [1.0, 2.0, 3.0]
        out2_copy = copy(out2)
        apply_hcv_correction!(out2, hcv2, x, anchor, hvp_mock!, s, hvp_buf)
        @test out2 == out2_copy
    end

    @testset "Cauchy damping behavior" begin
        hcv = HCVState(2)
        hcv.C_matrix .= I(2)
        hcv.vstar2 = 1.0
        hcv.enabled = true

        anchor = zeros(2)
        hvp_buf = zeros(2)
        hvp_mock! = (out, v) -> (out .= v; out)

        # Near anchor: α ≈ 1, correction ≈ full
        x_near = [0.001, 0.001]
        v_near = x_near
        vnorm2_near = sum(abs2, v_near)
        α_near = 1.0 / (1.0 + vnorm2_near)
        @test α_near > 0.999

        # Far from anchor: α ≈ 0
        x_far = [100.0, 100.0]
        v_far = x_far
        vnorm2_far = sum(abs2, v_far)
        α_far = 1.0 / (1.0 + vnorm2_far)
        @test α_far < 0.0001

        # Verify via actual correction
        out_near = zeros(2)
        apply_hcv_correction!(out_near, hcv, x_near, anchor, hvp_mock!, 1.0, hvp_buf)
        out_far = zeros(2)
        apply_hcv_correction!(out_far, hcv, x_far, anchor, hvp_mock!, 1.0, hvp_buf)

        # Per-unit-displacement correction should be much smaller far away
        @test norm(out_far) / norm(v_far) < 0.01 * norm(out_near) / norm(v_near)
    end
end

@testset "AnchorBank" begin
    d = 4

    @testset "construction" begin
        bank = AnchorBank(d; capacity=5)
        @test length(bank.entries) == 5
        @test bank.n_populated == 0
        @test bank.active_idx == 0
        @test !has_active_anchor(bank)
    end

    @testset "add_anchor! and active_entry" begin
        bank = AnchorBank(d; capacity=3)

        pos1 = [1.0, 0.0, 0.0, 0.0]
        grad1 = [0.1, 0.2, 0.3, 0.4]

        idx = add_anchor!(bank; position=pos1, full_gradient=grad1)
        @test idx == 1
        @test bank.n_populated == 1
        @test has_active_anchor(bank)
        @test active_entry(bank).position == pos1
        @test active_entry(bank).full_gradient == grad1
    end

    @testset "select_nearest!" begin
        bank = AnchorBank(d; capacity=5)

        # Add 3 anchors at distinct locations
        add_anchor!(bank; position=[0.0, 0.0, 0.0, 0.0], full_gradient=zeros(d))
        add_anchor!(bank; position=[10.0, 0.0, 0.0, 0.0], full_gradient=zeros(d))
        add_anchor!(bank; position=[5.0, 5.0, 0.0, 0.0], full_gradient=zeros(d))

        # Point near origin should select anchor 1
        entry = select_nearest!(bank, [0.1, 0.1, 0.0, 0.0])
        @test bank.active_idx == 1
        @test entry.position ≈ [0.0, 0.0, 0.0, 0.0]

        # Point near [10,0,0,0] should select anchor 2
        entry = select_nearest!(bank, [9.5, 0.0, 0.0, 0.0])
        @test bank.active_idx == 2

        # Point near [5,5,0,0] should select anchor 3
        entry = select_nearest!(bank, [5.1, 4.9, 0.0, 0.0])
        @test bank.active_idx == 3
    end

    @testset "select_nearest! empty bank" begin
        bank = AnchorBank(d; capacity=5)
        @test select_nearest!(bank, [1.0, 2.0, 3.0, 4.0]) === nothing
    end

    @testset "LRU eviction" begin
        bank = AnchorBank(d; capacity=2)

        add_anchor!(bank; position=[0.0, 0.0, 0.0, 0.0], full_gradient=zeros(d))
        add_anchor!(bank; position=[10.0, 0.0, 0.0, 0.0], full_gradient=zeros(d))

        # Use anchor 1 repeatedly (resets its age), anchor 2 ages
        for _ in 1:5
            select_nearest!(bank, [0.1, 0.0, 0.0, 0.0])
        end

        # Anchor 2 should have higher age
        @test bank.entries[2].age > bank.entries[1].age

        # Adding a 3rd anchor should evict anchor 2 (highest age)
        new_pos = [20.0, 0.0, 0.0, 0.0]
        idx = add_anchor!(bank; position=new_pos, full_gradient=zeros(d))
        @test idx == 2
        @test bank.entries[2].position == new_pos
        @test bank.n_populated == 2
    end

    @testset "age tracking" begin
        bank = AnchorBank(d; capacity=3)

        add_anchor!(bank; position=[0.0, 0.0, 0.0, 0.0], full_gradient=zeros(d))
        add_anchor!(bank; position=[10.0, 0.0, 0.0, 0.0], full_gradient=zeros(d))

        # All ages start at 0
        @test bank.entries[1].age == 0
        @test bank.entries[2].age == 0

        # Selecting anchor 1 increments anchor 2's age, resets anchor 1's
        select_nearest!(bank, [0.0, 0.0, 0.0, 0.0])
        @test bank.entries[1].age == 0
        @test bank.entries[2].age == 1

        select_nearest!(bank, [0.0, 0.0, 0.0, 0.0])
        @test bank.entries[1].age == 0
        @test bank.entries[2].age == 2
    end

    @testset "construction with HCV" begin
        bank = AnchorBank(d; capacity=3, use_hcv=true)
        @test bank.entries[1].hcv isa HCVState
        @test !bank.entries[1].hcv.enabled
    end

    @testset "empty!" begin
        bank = AnchorBank(d; capacity=3)
        add_anchor!(bank; position=ones(d), full_gradient=zeros(d))
        add_anchor!(bank; position=2 .* ones(d), full_gradient=zeros(d))
        @test bank.n_populated == 2
        @test has_active_anchor(bank)

        empty!(bank)
        @test bank.n_populated == 0
        @test isempty(bank)
        @test bank.active_idx == 0
        @test !has_active_anchor(bank)
    end
end

@testset "Triangle-prune selection" begin
    d = 4

    @testset "parity with bruteforce (random)" begin
        Random.seed!(42)
        anchors = [randn(d) for _ in 1:15]
        queries = [randn(d) for _ in 1:50]

        for K in [3, 10, 15]
            bank_bf = AnchorBank(d; capacity=K, strategy=:bruteforce)
            bank_tp = AnchorBank(d; capacity=K, strategy=:triangle_prune)
            for a in anchors[1:K]
                g = randn(d)
                add_anchor!(bank_bf; position=copy(a), full_gradient=copy(g))
                add_anchor!(bank_tp; position=copy(a), full_gradient=copy(g))
            end
            for q in queries
                entry_bf = select_nearest!(bank_bf, q)
                entry_tp = select_nearest!(bank_tp, q)
                @test entry_bf.position == entry_tp.position
                @test bank_bf.active_idx == bank_tp.active_idx
            end
        end
    end

    @testset "age tracking matches bruteforce" begin
        bank_bf = AnchorBank(d; capacity=5, strategy=:bruteforce)
        bank_tp = AnchorBank(d; capacity=5, strategy=:triangle_prune)
        positions = [[1.0, 0, 0, 0], [0, 10.0, 0, 0], [5.0, 5.0, 0, 0]]
        for p in positions
            add_anchor!(bank_bf; position=copy(p), full_gradient=zeros(d))
            add_anchor!(bank_tp; position=copy(p), full_gradient=zeros(d))
        end

        queries = [[0.1, 0.1, 0, 0], [9.0, 0, 0, 0], [5.0, 5.0, 0, 0], [0, 0, 0, 0]]
        for q in queries
            select_nearest!(bank_bf, q)
            select_nearest!(bank_tp, q)
            for k in 1:bank_bf.n_populated
                @test bank_bf.entries[k].age == bank_tp.entries[k].age
            end
        end
    end

    @testset "inter_dist matrix integrity" begin
        bank = AnchorBank(d; capacity=5, strategy=:triangle_prune)
        p1 = [0.0, 0.0, 0.0, 0.0]
        p2 = [3.0, 4.0, 0.0, 0.0]
        p3 = [1.0, 0.0, 0.0, 0.0]

        add_anchor!(bank; position=copy(p1), full_gradient=zeros(d))
        add_anchor!(bank; position=copy(p2), full_gradient=zeros(d))
        add_anchor!(bank; position=copy(p3), full_gradient=zeros(d))

        # Diagonal is zero
        for k in 1:bank.n_populated
            @test bank.inter_dist[k, k] == 0.0
        end
        # Symmetry
        for i in 1:bank.n_populated, j in 1:bank.n_populated
            @test bank.inter_dist[i, j] == bank.inter_dist[j, i]
        end
        # Exact values
        @test bank.inter_dist[1, 2] ≈ 5.0  # sqrt(9+16)
        @test bank.inter_dist[1, 3] ≈ 1.0
        @test bank.inter_dist[2, 3] ≈ sqrt(4.0 + 16.0)
    end

    @testset "inter_dist after eviction" begin
        bank = AnchorBank(d; capacity=2, strategy=:triangle_prune)
        add_anchor!(bank; position=[0.0, 0, 0, 0], full_gradient=zeros(d))
        add_anchor!(bank; position=[10.0, 0, 0, 0], full_gradient=zeros(d))
        @test bank.inter_dist[1, 2] ≈ 10.0

        # Age anchor 2 so it gets evicted
        for _ in 1:5
            select_nearest!(bank, [0.1, 0, 0, 0])
        end

        # Evict anchor 2 by adding a new one
        add_anchor!(bank; position=[3.0, 4.0, 0, 0], full_gradient=zeros(d))
        @test bank.inter_dist[1, 2] ≈ 5.0  # updated after overwrite
        @test bank.inter_dist[2, 1] ≈ 5.0
    end

    @testset "single anchor fallback" begin
        bank = AnchorBank(d; capacity=5, strategy=:triangle_prune)
        add_anchor!(bank; position=[1.0, 2.0, 3.0, 4.0], full_gradient=zeros(d))
        entry = select_nearest!(bank, zeros(d))
        @test entry.position == [1.0, 2.0, 3.0, 4.0]
        @test bank.active_idx == 1
    end

    @testset "empty bank" begin
        bank = AnchorBank(d; capacity=5, strategy=:triangle_prune)
        @test select_nearest!(bank, ones(d)) === nothing
    end
end

@testset "AnchorBankAdapter" begin
    @testset "construction" begin
        select_calls = Ref(0)
        update_calls = Ref(0)
        select_fn! = x -> (select_calls[] += 1)
        update_fn! = trace -> (update_calls[] += 1)

        adapter = AnchorBankAdapter(select_fn!, update_fn!, 10.0, 0.0, true)
        @test adapter.update_dt == 10.0
        @test adapter.warmup_only
    end

    @testset "type stability" begin
        select_fn! = x -> nothing
        update_fn! = trace -> nothing
        adapter = AnchorBankAdapter(select_fn!, update_fn!, 10.0, 0.0, true)
        @test adapter isa AnchorBankAdapter{typeof(select_fn!), typeof(update_fn!)}
    end

    @testset "adapt! calls select every step" begin
        select_positions = Vector{Float64}[]
        update_traces = []
        select_fn! = x -> push!(select_positions, copy(x))
        update_fn! = trace -> push!(update_traces, trace)

        adapter = AnchorBankAdapter(select_fn!, update_fn!, 100.0, 0.0, true)

        d = 3
        flow = ZigZag(d)
        grad = SubsampledGradient(
            (out, x) -> (out .= x; out), n -> nothing, 10
        )
        x_test = [1.0, 2.0, 3.0]
        state = PDMPSamplers.PDMPState(5.0, SkeletonPoint(x_test, ones(d)))
        trace_mgr = PDMPSamplers.TraceManager(state, flow, GridThinningStrategy(), 100.0)

        PDMPSamplers.adapt!(adapter, state, flow, grad, trace_mgr; phase=:warmup)

        @test length(select_positions) == 1
        @test select_positions[1] == x_test
        @test isempty(update_traces)  # update_dt=100, t=5, not triggered
    end
end

@testset "default_adapter with AnchorBankAdapter" begin
    d = 3
    select_fn! = x -> nothing
    update_fn! = trace -> nothing
    bank_adapter = AnchorBankAdapter(select_fn!, update_fn!, 0.0, 0.0, true)

    flow = ZigZag(d)
    grad = SubsampledGradient(
        (out, x) -> (out .= x; out), n -> nothing,
        trace -> nothing, (out, x) -> (out .= x; out),
        10, 5, true; resample_dt=2.0
    )

    adapter = PDMPSamplers.default_adapter(flow, grad, bank_adapter, 10.0, 50.0, 0.0)
    @test adapter isa SequenceAdapter
    @test bank_adapter.update_dt == 10.0  # t_warmup / no_anchor_updates = 50 / 5
    @test bank_adapter.last_update == 0.0
end
