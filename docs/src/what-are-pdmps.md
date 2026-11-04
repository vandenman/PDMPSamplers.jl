# What are PDMPs?

Most MCMC samplers produce a sequence of draws: propose a new point, then
accept or reject it. A piecewise deterministic Markov process (PDMP) works
differently. It is a particle that moves continuously through parameter space
and changes direction at random times. Its path, run long enough, is
distributed according to the posterior.

This page explains the idea with pictures and a small example. You don't need
it to use the package, but it explains the options you'll meet later.

## A particle with a velocity

The state of a PDMP is a position ``x`` and a velocity ``v``. Between events,
the particle follows a fixed path: a straight line for the Zig-Zag and Bouncy
Particle samplers, an ellipse for the Boomerang. Write the posterior as
``\pi(x) \propto \exp(-U(x))``, so ``U`` is the negative log posterior. Events
happen at rate

```math
\lambda(x, v) = \max\{0,\ \langle v, \nabla U(x) \rangle\}.
```

The rate is zero while the particle moves toward higher posterior density, and
grows as it climbs away from it. At an event the velocity changes: the
Zig-Zag flips the sign of one component, and the Bouncy Particle sampler
reflects the velocity off the gradient, like a ball bouncing off a wall. That
is all it takes for the path to have the posterior as its stationary
distribution.

The target is a bivariate normal with correlation 0.7. Each sampler runs for
15 time units from the same starting point, and `PDMPDiscretize` reads off the
position every 0.01 time units so the path can be drawn:

```@example pdmp
using PDMPSamplers, Distributions, LinearAlgebra, ADTypes
import ForwardDiff

Σ = [1.0 0.7; 0.7 1.0]
target = MvNormal(zeros(2), Σ)
model = PDMPModel(2, LogDensity(x -> logpdf(target, x)), ADTypes.AutoForwardDiff(), true)
alg = GridThinningStrategy()
x0 = [-1.5, 1.5]

function sample_path(flow, model, alg, x0, T; seed=1)
    chains = pdmp_sample(copy(x0), flow, model, alg, 0.0, T; seed=seed, progress=false)
    return Matrix(PDMPDiscretize(chains, 0.01))   # one row per 0.01 time units
end

paths = [sample_path(ZigZag(2), model, alg, x0, 15.0),
         sample_path(BouncyParticle(2, 0.5), model, alg, x0, 15.0),
         sample_path(Boomerang(2, 0.5), model, alg, x0, 15.0)]
nothing # hide
```

Plot each path as a line, for example with Plots.jl:
`plot(paths[1][:, 1], paths[1][:, 2])`. This page draws them with a small
helper that writes SVG, so the documentation needs no plotting package.

::: details The plotting helper

```@example pdmp
# Draw trajectories as inline SVG; colours follow the site theme.
function trajectory_svg(paths, titles; lim=3.0, px=230, ellipse=nothing, show_axes=false)
    pad = 24
    sc(v) = pad + (v + lim) / (2lim) * (px - 2pad)
    sy(v) = px - sc(v)
    io = IOBuffer()
    columns = length(paths)
    max_width = columns * px + (columns - 1) * 12
    print(io, """<div style="display:grid;grid-template-columns:repeat($columns,minmax(0,1fr));gap:12px;max-width:$(max_width)px;margin:1em auto">""")
    for (P, title) in zip(paths, titles)
        print(io, """<svg viewBox="0 0 $px $(px + 18)" width="$px" style="display:block;width:100%;height:auto;min-width:0" role="img" aria-label="$title trajectory">""")
        print(io, """<rect x="$pad" y="$pad" width="$(px - 2pad)" height="$(px - 2pad)" fill="none" stroke="var(--vp-c-divider)"/>""")
        if show_axes
            print(io, """<line x1="$(sc(0))" y1="$pad" x2="$(sc(0))" y2="$(px - pad)" stroke="var(--vp-c-text-3)" stroke-dasharray="3 3"/>""")
            print(io, """<line x1="$pad" y1="$(sy(0))" x2="$(px - pad)" y2="$(sy(0))" stroke="var(--vp-c-text-3)" stroke-dasharray="3 3"/>""")
        end
        if ellipse !== nothing
            L = cholesky(ellipse).L
            for r in (1.0, 2.0)
                pts = [r * L * [cos(a), sin(a)] for a in range(0, 2π; length=100)]
                print(io, """<polygon fill="none" stroke="var(--vp-c-text-3)" stroke-width="0.8" points=\"""",
                    join(("$(sc(p[1])),$(sy(p[2]))" for p in pts), " "), "\"/>")
            end
        end
        print(io, """<polyline fill="none" stroke="var(--vp-c-brand-1)" stroke-width="1.4" stroke-linejoin="round" points=\"""",
            join(("$(round(sc(P[i, 1]); digits=1)),$(round(sy(P[i, 2]); digits=1))" for i in axes(P, 1)), " "), "\"/>")
        print(io, """<circle cx="$(sc(P[1, 1]))" cy="$(sy(P[1, 2]))" r="3.5" fill="var(--vp-c-text-1)"/>""")
        print(io, """<text x="$(px / 2)" y="$(px + 10)" text-anchor="middle" font-size="13" fill="var(--vp-c-text-1)">$title</text>""")
        print(io, "</svg>")
    end
    print(io, "</div>")
    return HTML(String(take!(io)))
end
nothing # hide
```

:::

```@example pdmp
trajectory_svg(paths, ["Zig-Zag", "Bouncy Particle", "Boomerang"]; ellipse=Σ)
```

The black dot is the start, and the grey ellipses contain 39% and 86% of the
posterior mass.

The Zig-Zag only moves diagonally, at unit speed in every coordinate. The
Bouncy Particle sampler moves in any direction, and every so often draws a
fresh velocity (a *refreshment*) so it can't get stuck on a closed loop. The
Boomerang moves on ellipses of a Gaussian reference distribution, here the
standard normal; events correct for the difference between that reference and
the posterior.

## The output is a path

A PDMP returns the whole path, stored as the positions and velocities at the
events. A posterior expectation is a time average along the path,

```math
\mathbb{E}_\pi[f(x)] \approx \frac{1}{T} \int_0^T f(x_t)\, \mathrm{d}t,
```

and for means, variances and inclusion probabilities PDMPSamplers computes
these integrals exactly, segment by segment. There is no step size, and so no
discretization error. The only error is the usual Monte Carlo error, which
shrinks as ``T`` grows.

```@example pdmp
chains = pdmp_sample(2, ZigZag(2), model, alg, 0.0, 10_000.0; progress=false)
(mean = mean(chains), cov = cov(chains))
```

For plots or code that expect draws, take the position at regular times with
`PDMPDiscretize(chains, dt)`, as for the pictures above.

## From MCMC to PDMPs

If you know MCMC, most concepts carry over:

| MCMC | PDMP |
| --- | --- |
| number of iterations | run time `T` |
| draws | the path; `PDMPDiscretize` gives draws at regular times |
| proposal and accept/reject step | none; the particle only changes direction at events |
| step size, number of leapfrog steps | none |
| warmup / burn-in | warmup time (7th argument of `pdmp_sample`) |
| effective sample size | `ess(chains)`, computed from the path |

## Simulating the event times

The hard part of a PDMP is simulating the next event time: the rate changes
along the path, and depends on the gradient. PDMPSamplers uses *thinning*. It
proposes times from a simpler process whose rate is an upper bound on
``\lambda``, evaluates the gradient at the proposed time, and accepts with
probability ``\lambda / \text{bound}``. `GridThinningStrategy` builds this
bound automatically from gradient and curvature evaluations along the path,
so you only provide the gradient (or let automatic differentiation do it).

## Why PDMPs?

- **Persistent motion.** The particle keeps its direction until an event, so it
  crosses the posterior in long straight runs instead of a random walk.
- **Continuous-time output**, as described above.
- **Variable selection** (below).
- **Subsampling** (below).

### Variable selection

Because the particle moves continuously, it passes through zero exactly. A
*sticky* PDMP can then hold a coefficient at zero for a random time: during
that time the coefficient is excluded from the model. So the sampler moves
between models of different sizes without proposals between models, and the
fraction of time a coefficient spends away from zero is its posterior
inclusion probability.

Below, a sticky Zig-Zag samples a two-dimensional spike-and-slab posterior.
The flat stretches on the dashed axes are periods where one coefficient is
excluded.

```@example pdmp
μ = [1.0, 0.4]
slab = MvNormal(μ, I(2))
sticky_model = PDMPModel(2, LogDensity(x -> logpdf(slab, x)), ADTypes.AutoForwardDiff(), true)
κ = [pdf(Normal(μ[i], 1), 0.0) for i in 1:2]       # prior inclusion probability 0.5
sticky_alg = Sticky(GridThinningStrategy(), κ)
sticky_path = sample_path(ZigZag(2), sticky_model, sticky_alg, [1.5, 1.5], 25.0; seed=3)
trajectory_svg([sticky_path], ["Sticky Zig-Zag"]; show_axes=true)
```

The [Variable selection](variable-selection.md) tutorial covers this in
detail.

### Subsampling

The event rate depends on the gradient, which for ``N`` observations is a sum
of ``N`` terms. A PDMP stays exact if it replaces the gradient at a proposed
event by an unbiased estimate from a few observations, as long as the bound
used for thinning covers every observation. Each event then costs a handful
of observations instead of the whole data set, while the output still targets
the exact posterior. Samplers such as Hamiltonian Monte Carlo cannot use
subsampled gradients without biasing the result. See
[Subsampling](subsampling.md).

## Further reading

- J. Bierkens, P. Fearnhead and G. Roberts (2019). The Zig-Zag process and
  super-efficient sampling for Bayesian analysis of big data. *The Annals of
  Statistics*, 47(3).
- A. Bouchard-Côté, S. J. Vollmer and A. Doucet (2018). The Bouncy Particle
  Sampler: a nonreversible rejection-free Markov chain Monte Carlo method.
  *Journal of the American Statistical Association*, 113(522).
- J. Bierkens, S. Grazzi, K. Kamatani and G. Roberts (2020). The Boomerang
  sampler. *Proceedings of the 37th International Conference on Machine
  Learning*.
- J. Bierkens, S. Grazzi, F. van der Meulen and M. Schauer (2023). Sticky PDMP
  samplers for sparse and local inference problems. *Statistics and
  Computing*, 33.
