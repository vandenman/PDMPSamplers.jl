# Sampling

[R function reference](../reference.md)

Generated from the PDMPSamplersR help files. The R package is the source of truth.

```@raw html
<span id="r-pdmp_sample"></span>
```

## `pdmp_sample`

```@raw html
<h3>Description</h3>

<p>Performs Piecewise Deterministic Markov Process (PDMP) sampling using the
PDMPSamplers.jl Julia package.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>pdmp_sample(
  f,
  d,
  flow = c("ZigZag", "BouncyParticle", "Boomerang", "AdaptiveBoomerang",
    "PreconditionedZigZag", "PreconditionedBPS"),
  algorithm = c("ThinningStrategy", "GridThinningStrategy", "RootsPoissonStrategy"),
  T = 50000,
  t0 = 0,
  t_warmup = 0,
  flow_mean = NULL,
  flow_cov = NULL,
  c0 = 0.01,
  x0 = NULL,
  theta0 = NULL,
  hessian = NULL,
  sticky = FALSE,
  can_stick = NULL,
  model_prior = NULL,
  parameter_prior = NULL,
  slab_prior = NULL,
  grid_n = 30,
  grid_t_max = 2,
  grid_bound = c("value_quadratic", "constant"),
  grid_curvature_bound = NULL,
  show_progress = TRUE,
  n_chains = 1L,
  threaded = FALSE,
  seed = NULL,
  adaptive_scheme = c("diagonal", "fullrank"),
  refresh_rate = 1,
  materialize = TRUE,
  trace_storage = c("dense", "streaming"),
  streaming_trace_directory = NULL,
  streaming_trace_buffer_events = 4096L,
  online_summaries = NULL,
  support_boundary = support_boundary_control(),
  control = sampler_control()
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="f">f</code></td>
<td>
<p>Function that computes the NEGATIVE gradient. Should take a numeric vector
and return a numeric vector of the same length.</p>
</td></tr>
<tr><td><code id="d">d</code></td>
<td>
<p>Integer, dimension of the problem.</p>
</td></tr>
<tr><td><code id="flow">flow</code></td>
<td>
<p>Character string specifying the flow type. One of &quot;ZigZag&quot;,
&quot;BouncyParticle&quot;, &quot;Boomerang&quot;, &quot;AdaptiveBoomerang&quot;, &quot;PreconditionedZigZag&quot;,
or &quot;PreconditionedBPS&quot;.
The '&quot;AdaptiveBoomerang&quot;' flow learns its reference (mean and precision)
during warmup and requires a grid-like algorithm. It is supported with
dependent slabs when 'GridThinningStrategy' is selected; provide positive
't_warmup', or let the sampler choose its 20
The '&quot;PreconditionedZigZag&quot;' and '&quot;PreconditionedBPS&quot;' flows learn a
diagonal preconditioner during warmup and also require
a grid-like algorithm.</p>
</td></tr>
<tr><td><code id="algorithm">algorithm</code></td>
<td>
<p>Character string specifying the algorithm. One of
&quot;ThinningStrategy&quot;, &quot;GridThinningStrategy&quot;, or &quot;RootsPoissonStrategy&quot;.</p>
</td></tr>
<tr><td><code id="T">T</code></td>
<td>
<p>Numeric, length of the main run in PDMP time units (default: 50000).
The main run starts after the warmup, and only it is used in summaries.</p>
</td></tr>
<tr><td><code id="t0">t0</code></td>
<td>
<p>Numeric, initial time (default: 0.0).</p>
</td></tr>
<tr><td><code id="t_warmup">t_warmup</code></td>
<td>
<p>Numeric, length of the warmup, which runs before the main
run (default: 0.0). Events during warmup are discarded. The run covers
time <code>t0</code> to <code>t0 + t_warmup + T</code>.</p>
</td></tr>
<tr><td><code id="flow_mean">flow_mean</code></td>
<td>
<p>Numeric vector of length d, mean vector for the flow (default: zero vector).</p>
</td></tr>
<tr><td><code id="flow_cov">flow_cov</code></td>
<td>
<p>Numeric matrix of size d x d, covariance matrix for the flow (default: identity matrix).</p>
</td></tr>
<tr><td><code id="c0">c0</code></td>
<td>
<p>Numeric, bound parameter (default: 1e-2).</p>
</td></tr>
<tr><td><code id="x0">x0</code></td>
<td>
<p>Numeric vector of length d, initial position (default: random normal).</p>
</td></tr>
<tr><td><code id="theta0">theta0</code></td>
<td>
<p>Numeric vector of length d, initial velocity (default: random based on the flow).</p>
</td></tr>
<tr><td><code id="hessian">hessian</code></td>
<td>
<p>Function that returns the negative Hessian matrix of size d x d (default: NULL).
Only used with GridThinningStrategy to compute the Hessian-vector product.</p>
</td></tr>
<tr><td><code id="sticky">sticky</code></td>
<td>
<p>Logical, whether to use sticky sampling (default: FALSE).</p>
</td></tr>
<tr><td><code id="can_stick">can_stick</code></td>
<td>
<p>Logical vector of length d, which coordinates can stick (default: all FALSE).</p>
</td></tr>
<tr><td><code id="model_prior">model_prior</code></td>
<td>
<p>Prior distribution object for model selection:
<code>bernoulli()</code> or <code>betabernoulli()</code>.</p>
</td></tr>
<tr><td><code id="parameter_prior">parameter_prior</code></td>
<td>
<p>Numeric vector of length d, prior parameters for sticky sampling (default: NULL).</p>
</td></tr>
<tr><td><code id="slab_prior">slab_prior</code></td>
<td>
<p>Optional dependent slab prior created by
<code>dense_gaussian_slab()</code>, <code>exchangeable_gaussian_slab()</code>,
or <code>loglinear_gaussian_scale_slab()</code>. Mutually exclusive with
<code>parameter_prior</code>. The full target supplied through <code>f</code> must
contain the complete coherent slab represented by <code>slab_prior</code>; the
package replaces that slab by its active-face restriction internally.</p>
</td></tr>
<tr><td><code id="grid_n">grid_n</code></td>
<td>
<p>Integer, number of grid points for GridThinningStrategy (default: 30).</p>
</td></tr>
<tr><td><code id="grid_t_max">grid_t_max</code></td>
<td>
<p>Numeric, maximum time for grid in GridThinningStrategy (default: 2.0).</p>
</td></tr>
<tr><td><code id="grid_bound">grid_bound</code></td>
<td>
<p>Character string selecting the GridThinning proposal
envelope. <code>"value_quadratic"</code> (default) bounds each grid cell by
the larger endpoint rate plus <code>grid_curvature_bound * h^2 / 8</code>;
without <code>grid_curvature_bound</code> the cells use the ordinary
GridThinning construction. <code>"constant"</code> always uses the ordinary
construction.</p>
</td></tr>
<tr><td><code id="grid_curvature_bound">grid_curvature_bound</code></td>
<td>
<p>Optional nonnegative bound on the curvature of
the event rate over a grid cell, used by <code>grid_bound =
"value_quadratic"</code>. Too small a value invalidates the envelope.</p>
</td></tr>
<tr><td><code id="show_progress">show_progress</code></td>
<td>
<p>Logical, whether to show progress bar (default: TRUE).</p>
</td></tr>
<tr><td><code id="n_chains">n_chains</code></td>
<td>
<p>Integer, number of chains to run (default: 1).</p>
</td></tr>
<tr><td><code id="threaded">threaded</code></td>
<td>
<p>Logical, whether to run chains in parallel (default: FALSE).</p>
</td></tr>
<tr><td><code id="seed">seed</code></td>
<td>
<p>NULL (default) or a non-negative integer seed. The seed controls
Julia's sampler RNG and any generated default initial position when
<code>x0 = NULL</code>.</p>
</td></tr>
<tr><td><code id="adaptive_scheme">adaptive_scheme</code></td>
<td>
<p>Character string, adaptation scheme for AdaptiveBoomerang.
One of &quot;diagonal&quot; (default, O(d) per update) or &quot;fullrank&quot; (O(d^3) per update,
better for correlated targets). Ignored for other flow types. With sticky
sampling, &quot;fullrank&quot; releases frozen coordinates with the experimental
boundary-proposal law; &quot;diagonal&quot; uses the stored-velocity law.</p>
</td></tr>
<tr><td><code id="refresh_rate">refresh_rate</code></td>
<td>
<p>Refreshment rate of the <code>"BouncyParticle"</code> and
<code>"PreconditionedBPS"</code> flows (default 1). Ignored by the other flows.</p>
</td></tr>
<tr><td><code id="materialize">materialize</code></td>
<td>
<p>Logical. If <code>TRUE</code> (default), the chain skeleton is
immediately extracted from Julia into R so that <code>saveRDS()</code> /
<code>readRDS()</code> work without any extra steps. Set to <code>FALSE</code> to skip
the extraction and keep only the live Julia reference.
This can save time and memory if you don't need to save the result or if you plan to call <code>materialize()</code> manually later.</p>
</td></tr>
<tr><td><code id="trace_storage">trace_storage</code></td>
<td>
<p><code>"dense"</code> (default) keeps the event trace in
memory. <code>"streaming"</code> writes it in chunks to
<code>streaming_trace_directory</code>, so memory use does not grow with the
number of events; it is required for <code>online_summaries</code>.</p>
</td></tr>
<tr><td><code id="streaming_trace_directory">streaming_trace_directory</code></td>
<td>
<p>Directory for the streamed trace. Required
when <code>trace_storage = "streaming"</code>, ignored otherwise.</p>
</td></tr>
<tr><td><code id="streaming_trace_buffer_events">streaming_trace_buffer_events</code></td>
<td>
<p>Number of events buffered in memory
before a chunk is written (default 4096). Used only with
<code>trace_storage = "streaming"</code>.</p>
</td></tr>
<tr><td><code id="online_summaries">online_summaries</code></td>
<td>
<p><code>NULL</code> (default) or options from
<code>online_summary_options()</code>. Requires <code>trace_storage = "streaming"</code>;
the summaries are read with <code>read_online_summaries()</code>.</p>
</td></tr>
<tr><td><code id="support_boundary">support_boundary</code></td>
<td>
<p>A list created by <code>support_boundary_control()</code> that
controls support-boundary diagnostics and heuristic event/refresh recovery.</p>
</td></tr>
<tr><td><code id="control">control</code></td>
<td>
<p>Additional call-local settings from <code>sampler_control</code>.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A <code>pdmp_result</code> object. Use <code>mean</code>, <code>var</code>,
<code>quantile</code>, etc. for continuous-time estimators, or
<code>discretize</code> to obtain a sample matrix.
</p>
```

```@raw html
<span id="r-pdmp_sample_from_stanmodel"></span>
```

## `pdmp_sample_from_stanmodel`

```@raw html
<h3>Description</h3>

<p>Performs Piecewise Deterministic Markov Process (PDMP) sampling from a Stan model
using the PDMPSamplers.jl Julia package with BridgeStan integration.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>pdmp_sample_from_stanmodel(
  path_to_stanmodel,
  standata,
  subsampling = NULL,
  subsampling_anchor_updates = 0L,
  subsampling_use_anchor_bank = FALSE,
  subsampling_anchor_bank_capacity = 8L,
  subsampling_anchor_selection_policy = c("adaptive", "position"),
  subsampling_position_anchor_bank_spec = NULL,
  subsampling_main_anchor_refresh_distance = Inf,
  subsampling_require_no_main_full_data = FALSE,
  subsampling_diagnostic_window = 0,
  flow = c("ZigZag", "BouncyParticle", "Boomerang", "AdaptiveBoomerang",
    "PreconditionedZigZag", "PreconditionedBPS"),
  algorithm = c("ThinningStrategy", "GridThinningStrategy", "RootsPoissonStrategy"),
  T = 50000,
  t0 = 0,
  t_warmup = 0,
  warmup_adaptation_interval = NULL,
  boomerang_variance_floor_fraction = 0,
  flow_mean = NULL,
  flow_cov = NULL,
  c0 = 0.01,
  flow_specification = c("auto", "initial", "frozen"),
  x0 = NULL,
  theta0 = NULL,
  free0 = NULL,
  stored_velocity0 = NULL,
  sticky = FALSE,
  can_stick = NULL,
  model_prior = NULL,
  parameter_prior = NULL,
  slab_prior = NULL,
  grid_n = 30,
  grid_t_max = 2,
  curvature_backend = NULL,
  grid_bound = c("value_quadratic", "constant"),
  grid_curvature_bound = NULL,
  show_progress = TRUE,
  n_chains = 1L,
  threaded = FALSE,
  seed = NULL,
  adaptive_scheme = c("diagonal", "fullrank"),
  refresh_rate = 1,
  materialize = TRUE,
  trace_storage = c("dense", "streaming"),
  streaming_trace_directory = NULL,
  streaming_trace_buffer_events = 4096L,
  online_summaries = NULL,
  support_boundary = support_boundary_control(),
  control = sampler_control(),
  compile_options = stan_compile_options()
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="path_to_stanmodel">path_to_stanmodel</code></td>
<td>
<p>Character, path to a Stan model file (.stan) or a
compiled Stan model (.so/.dll/.dylib). If a .stan file is provided,
BridgeStan will compile it automatically.</p>
</td></tr>
<tr><td><code id="standata">standata</code></td>
<td>
<p>Either a character path to the Stan data file (JSON format),
or a named list that will be written to a temporary JSON file via
[write_stan_json()].</p>
</td></tr>
<tr><td><code id="subsampling">subsampling</code></td>
<td>
<p>Optional exact subsampling custom-Stan specification
created by [stan_subsampling()]. The selected Stan branch must add
the unscaled sum of the installed observations; Julia applies 'N / m'.</p>
</td></tr>
<tr><td><code id="subsampling_anchor_updates">subsampling_anchor_updates</code></td>
<td>
<p>Number of prepared OMRF anchors added from
continuous-time warmup means.</p>
</td></tr>
<tr><td><code id="subsampling_use_anchor_bank">subsampling_use_anchor_bank</code></td>
<td>
<p>Retain prepared OMRF anchors and select
the nearest entry at event boundaries. If false, updates replace the
active anchor.</p>
</td></tr>
<tr><td><code id="subsampling_anchor_bank_capacity">subsampling_anchor_bank_capacity</code></td>
<td>
<p>Maximum prepared OMRF anchors kept
per chain.</p>
</td></tr>
<tr><td><code id="subsampling_anchor_selection_policy">subsampling_anchor_selection_policy</code></td>
<td>
<p>Research anchor-bank policy.
'&quot;adaptive&quot;' preserves the existing event-history/envelope policy;
'&quot;position&quot;' selects one frozen finite bank source by weighted position
and supports linear Zig&ndash;Zag/BPS and harmonic Boomerang flows.</p>
</td></tr>
<tr><td><code id="subsampling_position_anchor_bank_spec">subsampling_position_anchor_bank_spec</code></td>
<td>
<p>Optional version-1 mathematical
position-bank specification saved on a previous result. Restoration
rebuilds prepared provider quantities from its ordered positions and
frozen distance scales; it requires zero warmup and matching dataset,
parameter-order, and envelope/grouping identities.</p>
</td></tr>
<tr><td><code id="subsampling_main_anchor_refresh_distance">subsampling_main_anchor_refresh_distance</code></td>
<td>
<p>Positive Euclidean distance
from the nearest prepared anchor that triggers a new exact OMRF anchor
during main sampling. The default 'Inf' disables main-phase preparation.</p>
</td></tr>
<tr><td><code id="subsampling_require_no_main_full_data">subsampling_require_no_main_full_data</code></td>
<td>
<p>If 'TRUE', fail the run when
retained sampling performs any full-gradient evaluation, prepares an
anchor, or traverses every OMRF person-node conditional in native Julia.
Setup and warmup work are allowed. This is intended as a hard benchmark
invariant for frozen-control-variate and direct-clock experiments.</p>
</td></tr>
<tr><td><code id="subsampling_diagnostic_window">subsampling_diagnostic_window</code></td>
<td>
<p>Width, in sampling time, of the windows
over which the OMRF anchor manager records retained-phase subsampling
diagnostics. The default 0 disables them.</p>
</td></tr>
<tr><td><code id="flow">flow</code></td>
<td>
<p>Character string specifying the flow type. One of &quot;ZigZag&quot;,
&quot;BouncyParticle&quot;, &quot;Boomerang&quot;, &quot;AdaptiveBoomerang&quot;, &quot;PreconditionedZigZag&quot;,
or &quot;PreconditionedBPS&quot;.
The '&quot;AdaptiveBoomerang&quot;' flow learns its reference (mean and precision)
during warmup and requires a grid-like algorithm. It is supported with
dependent slabs when 'GridThinningStrategy' is selected; provide positive
't_warmup', or let the sampler choose its 20
The '&quot;PreconditionedZigZag&quot;' and '&quot;PreconditionedBPS&quot;' flows learn a
diagonal preconditioner during warmup and also require
a grid-like algorithm.</p>
</td></tr>
<tr><td><code id="algorithm">algorithm</code></td>
<td>
<p>Character string specifying the algorithm. One of
&quot;ThinningStrategy&quot;, &quot;GridThinningStrategy&quot;, or &quot;RootsPoissonStrategy&quot;.</p>
</td></tr>
<tr><td><code id="T">T</code></td>
<td>
<p>Numeric, length of the main run in PDMP time units (default: 50000).
The main run starts after the warmup, and only it is used in summaries.</p>
</td></tr>
<tr><td><code id="t0">t0</code></td>
<td>
<p>Numeric, initial time (default: 0.0).</p>
</td></tr>
<tr><td><code id="t_warmup">t_warmup</code></td>
<td>
<p>Numeric, length of the warmup, which runs before the main
run (default: 0.0). Events during warmup are discarded. The run covers
time <code>t0</code> to <code>t0 + t_warmup + T</code>.</p>
</td></tr>
<tr><td><code id="warmup_adaptation_interval">warmup_adaptation_interval</code></td>
<td>
<p>Nonnegative time between dynamics
adaptation attempts during warmup. 'NULL' uses ten equally spaced warmup
windows ('t_warmup / 10'); zero attempts adaptation at every event. The
same interval is used by full-gradient and subsampling runs.</p>
</td></tr>
<tr><td><code id="boomerang_variance_floor_fraction">boomerang_variance_floor_fraction</code></td>
<td>
<p>Nonnegative opt-in research
safeguard for adaptive diagonal Boomerang flows. At every adaptation
update, named 'global_scale' and 'node_scale.*' coordinates retain at
least this fraction of their preadaptation variance. Zero (the default)
disables the safeguard.</p>
</td></tr>
<tr><td><code id="flow_mean">flow_mean</code></td>
<td>
<p>Numeric vector of length d, mean vector for the flow (default: zero vector).</p>
</td></tr>
<tr><td><code id="flow_cov">flow_cov</code></td>
<td>
<p>Numeric matrix of size d x d, covariance matrix for the flow (default: identity matrix).</p>
</td></tr>
<tr><td><code id="c0">c0</code></td>
<td>
<p>Numeric, bound parameter (default: 1e-2).</p>
</td></tr>
<tr><td><code id="flow_specification">flow_specification</code></td>
<td>
<p>Role of an explicitly supplied 'flow_mean' or
'flow_cov'. '&quot;initial&quot;' permits legitimate warmup adaptation and records
every chain's retained flow. '&quot;frozen&quot;' requires every retained chain to
match the request and fails closed otherwise. '&quot;auto&quot;' uses '&quot;initial&quot;'
when 't_warmup &gt; 0' and '&quot;frozen&quot;' otherwise.</p>
</td></tr>
<tr><td><code id="x0">x0</code></td>
<td>
<p>Numeric vector of length d, initial position (default: random normal).</p>
</td></tr>
<tr><td><code id="theta0">theta0</code></td>
<td>
<p>Optional initial velocity forwarded to both full-gradient and
custom-Stan subsampling initialization.</p>
</td></tr>
<tr><td><code id="free0">free0</code>, <code id="stored_velocity0">stored_velocity0</code></td>
<td>
<p>Optional sticky state to continue a previous
run: <code>free0</code> is the logical mask of free coordinates and
<code>stored_velocity0</code> the stored velocities of the frozen ones. Both
must be supplied together, with length equal to the number of
parameters, and require <code>sticky = TRUE</code>.</p>
</td></tr>
<tr><td><code id="sticky">sticky</code></td>
<td>
<p>Logical, whether to use sticky sampling (default: FALSE).</p>
</td></tr>
<tr><td><code id="can_stick">can_stick</code></td>
<td>
<p>Logical vector of length d, which coordinates can stick (default: all FALSE).</p>
</td></tr>
<tr><td><code id="model_prior">model_prior</code></td>
<td>
<p>Prior distribution object for model selection:
<code>bernoulli()</code> or <code>betabernoulli()</code>.</p>
</td></tr>
<tr><td><code id="parameter_prior">parameter_prior</code></td>
<td>
<p>Numeric vector of length d, prior parameters for sticky sampling (default: NULL).</p>
</td></tr>
<tr><td><code id="slab_prior">slab_prior</code></td>
<td>
<p>Optional dependent slab prior created by
<code>dense_gaussian_slab()</code>, <code>exchangeable_gaussian_slab()</code>,
or <code>loglinear_gaussian_scale_slab()</code>. Mutually exclusive with
<code>parameter_prior</code>. The full target supplied through <code>f</code> must
contain the complete coherent slab represented by <code>slab_prior</code>; the
package replaces that slab by its active-face restriction internally.</p>
</td></tr>
<tr><td><code id="grid_n">grid_n</code></td>
<td>
<p>Integer, number of grid points for GridThinningStrategy (default: 30).</p>
</td></tr>
<tr><td><code id="grid_t_max">grid_t_max</code></td>
<td>
<p>Numeric, maximum time for grid in GridThinningStrategy (default: 2.0).</p>
</td></tr>
<tr><td><code id="curvature_backend">curvature_backend</code></td>
<td>
<p>Character string selecting directional curvature:
'&quot;exact&quot;' or '&quot;finite_difference&quot;'. Custom-Stan subsampling has no
deterministic exact HVP unless using the analytic dependent OMRF provider.
Finite differences use
shifted exact gradients but approximate their directional derivative and
therefore have step-size and truncation error. 'NULL' selects exact
curvature for full gradients and finite differences for subsampling,
except that analytic dependent OMRF always uses its exact composed HVP.</p>
</td></tr>
<tr><td><code id="grid_bound">grid_bound</code></td>
<td>
<p>Character string selecting the GridThinning proposal
envelope. <code>"value_quadratic"</code> (default) bounds each grid cell by
the larger endpoint rate plus <code>grid_curvature_bound * h^2 / 8</code>;
without <code>grid_curvature_bound</code> the cells use the ordinary
GridThinning construction. <code>"constant"</code> always uses the ordinary
construction.</p>
</td></tr>
<tr><td><code id="grid_curvature_bound">grid_curvature_bound</code></td>
<td>
<p>Optional nonnegative bound on the curvature of
the event rate over a grid cell, used by <code>grid_bound =
"value_quadratic"</code>. Too small a value invalidates the envelope.</p>
</td></tr>
<tr><td><code id="show_progress">show_progress</code></td>
<td>
<p>Logical, whether to show progress bar (default: TRUE).</p>
</td></tr>
<tr><td><code id="n_chains">n_chains</code></td>
<td>
<p>Integer, number of chains to run (default: 1).</p>
</td></tr>
<tr><td><code id="threaded">threaded</code></td>
<td>
<p>Logical, whether to run chains in parallel (default: FALSE).</p>
</td></tr>
<tr><td><code id="seed">seed</code></td>
<td>
<p>NULL (default) or a non-negative integer seed. The seed controls
Julia's sampler RNG and any generated default initial position when
<code>x0 = NULL</code>.</p>
</td></tr>
<tr><td><code id="adaptive_scheme">adaptive_scheme</code></td>
<td>
<p>Character string, adaptation scheme for AdaptiveBoomerang.
One of &quot;diagonal&quot; (default, O(d) per update) or &quot;fullrank&quot; (O(d^3) per update,
better for correlated targets). Ignored for other flow types. With sticky
sampling, &quot;fullrank&quot; releases frozen coordinates with the experimental
boundary-proposal law; &quot;diagonal&quot; uses the stored-velocity law.</p>
</td></tr>
<tr><td><code id="refresh_rate">refresh_rate</code></td>
<td>
<p>Refreshment rate of the <code>"BouncyParticle"</code> and
<code>"PreconditionedBPS"</code> flows (default 1). Ignored by the other flows.</p>
</td></tr>
<tr><td><code id="materialize">materialize</code></td>
<td>
<p>Logical. If <code>TRUE</code> (default), the chain skeleton is
immediately extracted from Julia into R so that <code>saveRDS()</code> /
<code>readRDS()</code> work without any extra steps. Set to <code>FALSE</code> to skip
the extraction and keep only the live Julia reference.
This can save time and memory if you don't need to save the result or if you plan to call <code>materialize()</code> manually later.</p>
</td></tr>
<tr><td><code id="trace_storage">trace_storage</code></td>
<td>
<p><code>"dense"</code> (default) keeps the event trace in
memory. <code>"streaming"</code> writes it in chunks to
<code>streaming_trace_directory</code>, so memory use does not grow with the
number of events; it is required for <code>online_summaries</code>.</p>
</td></tr>
<tr><td><code id="streaming_trace_directory">streaming_trace_directory</code></td>
<td>
<p>Directory for the streamed trace. Required
when <code>trace_storage = "streaming"</code>, ignored otherwise.</p>
</td></tr>
<tr><td><code id="streaming_trace_buffer_events">streaming_trace_buffer_events</code></td>
<td>
<p>Number of events buffered in memory
before a chunk is written (default 4096). Used only with
<code>trace_storage = "streaming"</code>.</p>
</td></tr>
<tr><td><code id="online_summaries">online_summaries</code></td>
<td>
<p><code>NULL</code> (default) or options from
<code>online_summary_options()</code>. Requires <code>trace_storage = "streaming"</code>;
the summaries are read with <code>read_online_summaries()</code>.</p>
</td></tr>
<tr><td><code id="support_boundary">support_boundary</code></td>
<td>
<p>A list created by <code>support_boundary_control()</code> that
controls support-boundary diagnostics and heuristic event/refresh recovery.</p>
</td></tr>
<tr><td><code id="control">control</code></td>
<td>
<p>Additional call-local settings from <code>sampler_control</code>.</p>
</td></tr>
<tr><td><code id="compile_options">compile_options</code></td>
<td>
<p>Settings from <code>stan_compile_options</code>.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>The gradient and Hessian-vector product are automatically derived from the Stan model
via the BridgeStan extension of PDMPSamplers.jl. This means you do not need to provide
a separate gradient or Hessian function.
</p>


<h3>Value</h3>

<p>A <code>pdmp_result</code> object. Use <code>mean</code>, <code>var</code>,
<code>quantile</code>, etc. for continuous-time estimators, or
<code>discretize</code> to obtain a sample matrix.
</p>
```

```@raw html
<span id="r-brm_pdmp"></span>
```

## `brm_pdmp`

```@raw html
<h3>Description</h3>

<p>Uses PDMPSamplers.jl as a sampling backend for brms models.
Returns a standard 'brmsfit' object with all post-processing
('summary', 'plot', 'conditional_effects', 'loo', etc.) working.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>brm_pdmp(
  formula,
  data,
  family = gaussian(),
  prior = NULL,
  flow = c("ZigZag", "BouncyParticle", "Boomerang", "AdaptiveBoomerang",
    "PreconditionedZigZag", "PreconditionedBPS", "DensePreconditionedZigZag",
    "DensePreconditionedBPS"),
  algorithm = c("GridThinningStrategy", "ThinningStrategy", "RootsPoissonStrategy"),
  adaptive_scheme = c("diagonal", "fullrank", "lowrank"),
  T = 50000,
  t0 = 0,
  t_warmup = 0,
  flow_mean = NULL,
  flow_cov = NULL,
  c0 = 0.01,
  grid_n = 30,
  grid_t_max = 2,
  show_progress = TRUE,
  discretize_dt = NULL,
  n_chains = 1L,
  threaded = FALSE,
  seed = NULL,
  compute_lp = FALSE,
  subsample_size = NULL,
  n_anchor_updates = 0L,
  use_hcv = FALSE,
  use_anchor_bank = FALSE,
  bank_capacity = 20L,
  curvature_backend = c("exact", "finite_difference"),
  sticky = FALSE,
  can_stick = NULL,
  model_prior = NULL,
  kappa = NULL,
  slab_prior = NULL,
  stanvars = NULL,
  sample_prior = "no",
  save_model = NULL,
  control = sampler_control(),
  compile_options = stan_compile_options(),
  ...
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="formula">formula</code></td>
<td>
<p>A brms model formula.</p>
</td></tr>
<tr><td><code id="data">data</code></td>
<td>
<p>A data frame containing the variables in the model.</p>
</td></tr>
<tr><td><code id="family">family</code></td>
<td>
<p>A family object (e.g., 'gaussian()', 'bernoulli()').</p>
</td></tr>
<tr><td><code id="prior">prior</code></td>
<td>
<p>A 'brmsprior' object or NULL for default priors.</p>
</td></tr>
<tr><td><code id="flow">flow</code></td>
<td>
<p>Character string specifying the PDMP flow type.</p>
</td></tr>
<tr><td><code id="algorithm">algorithm</code></td>
<td>
<p>Character string specifying the Poisson time strategy.</p>
</td></tr>
<tr><td><code id="adaptive_scheme">adaptive_scheme</code></td>
<td>
<p>Character string for AdaptiveBoomerang covariance
adaptation: '&quot;diagonal&quot;', '&quot;fullrank&quot;', or '&quot;lowrank&quot;'.</p>
</td></tr>
<tr><td><code id="T">T</code></td>
<td>
<p>Numeric length of the main run, which starts after the warmup.</p>
</td></tr>
<tr><td><code id="t0">t0</code></td>
<td>
<p>Numeric start time.</p>
</td></tr>
<tr><td><code id="t_warmup">t_warmup</code></td>
<td>
<p>Numeric warmup duration, run before the main run.
Auto-set for adaptive flows. When subsampling is active and 't_warmup'
is 0, it is automatically set to 20% of 'T'.</p>
</td></tr>
<tr><td><code id="flow_mean">flow_mean</code></td>
<td>
<p>Numeric vector for the flow reference mean, or NULL. For
observation subsampling this is also the fixed control-variate anchor and
initial position; supplying a posterior mode or other representative point
can materially tighten the residual envelope.</p>
</td></tr>
<tr><td><code id="flow_cov">flow_cov</code></td>
<td>
<p>Numeric matrix for the flow covariance, or NULL.</p>
</td></tr>
<tr><td><code id="c0">c0</code></td>
<td>
<p>Numeric thinning bound constant. With 'ThinningStrategy' this
user-configured bound must dominate the deterministic subsampling component;
violations are detected and stop the sampler.</p>
</td></tr>
<tr><td><code id="grid_n">grid_n</code></td>
<td>
<p>Integer number of grid points for GridThinningStrategy.</p>
</td></tr>
<tr><td><code id="grid_t_max">grid_t_max</code></td>
<td>
<p>Numeric max grid interval for GridThinningStrategy.</p>
</td></tr>
<tr><td><code id="show_progress">show_progress</code></td>
<td>
<p>Logical; show sampling progress bar.</p>
</td></tr>
<tr><td><code id="discretize_dt">discretize_dt</code></td>
<td>
<p>Numeric time step for discretization, or NULL
for automatic (yields ~1000 samples).</p>
</td></tr>
<tr><td><code id="n_chains">n_chains</code></td>
<td>
<p>Integer number of chains (default: 1). With multiple
chains, 'Rhat' and multi-chain diagnostics become available.</p>
</td></tr>
<tr><td><code id="threaded">threaded</code></td>
<td>
<p>Logical; run chains in parallel (default: FALSE).</p>
</td></tr>
<tr><td><code id="seed">seed</code></td>
<td>
<p>NULL (default) or a non-negative integer seed passed through to
Julia's sampler RNG.</p>
</td></tr>
<tr><td><code id="compute_lp">compute_lp</code></td>
<td>
<p>Logical; compute 'lp__' via 'BridgeStan::log_density()'
for each sample (default: FALSE). Adds overhead but enables
'bridge_sampler()' and populates the 'lp__' diagnostic column.</p>
</td></tr>
<tr><td><code id="subsample_size">subsample_size</code></td>
<td>
<p>Integer number of observations per subsample,
or NULL (default) for full-data gradients. With 'GridThinningStrategy'
or a representable 'ThinningStrategy' envelope,
subsampling acceleration is selected automatically when affine predictor,
first-batch likelihood, and trajectory providers are all available for
the requested dynamics. Ineligible models use exact full gradients.</p>
</td></tr>
<tr><td><code id="n_anchor_updates">n_anchor_updates</code></td>
<td>
<p>Integer number of anchor updates during warmup
(default: 0). For analytic subsampling sampling, each update prepares a
coherent anchor state containing the full and deterministic anchor
gradients, cached likelihood predictors, and residual envelope.</p>
</td></tr>
<tr><td><code id="use_hcv">use_hcv</code></td>
<td>
<p>Logical; enable the certified analytic damped Hessian
control variate for subsampling affine Bernoulli-logit or binomial-logit
sampling (default: FALSE). Its likelihood Hessian and Taylor-remainder
envelope are prepared with each anchor, with Cauchy damping
'd / (d + ||x - anchor||^2)' in 'd' unconstrained dimensions;
finite-difference HCV is not used.</p>
</td></tr>
<tr><td><code id="use_anchor_bank">use_anchor_bank</code></td>
<td>
<p>Logical; retain prepared warmup anchors and select
the nearest one at event boundaries (default: FALSE). Selection is
chain-local and performs no full-data gradient or envelope rebuild.
Requires 'n_anchor_updates &gt; 0'; use only 'n_anchor_updates' for the
lower-memory single-anchor update mode.</p>
</td></tr>
<tr><td><code id="bank_capacity">bank_capacity</code></td>
<td>
<p>Integer capacity of the anchor bank (default: 20).
Least-recently-used entries are replaced during warmup. Each populated
entry stores predictor and kernel caches of order 'N * K' plus a complete
residual envelope of order 'N * R'; memory therefore grows linearly in
the populated capacity and can be substantial for large 'N' or many
growth-envelope components. Benchmark representative models before using
a large capacity. Analytic HCV additionally stores one dense 'd * d'
likelihood Hessian per populated anchor.</p>
</td></tr>
<tr><td><code id="curvature_backend">curvature_backend</code></td>
<td>
<p>Directional curvature evaluation: '&quot;exact&quot;'
(default) uses Hessian-vector products; '&quot;finite_difference&quot;' uses shifted
gradients and has finite-difference truncation error.</p>
</td></tr>
<tr><td><code id="sticky">sticky</code></td>
<td>
<p>Logical; enable spike-and-slab variable selection for
population-level coefficients (default: FALSE). Requires
'model_prior' to be set. Sticky transitions redraw the complete velocity
from the exact law on each active coordinate stratum; no pre-freeze
velocity is restored. This applies to every supported dynamics, including
dense-preconditioned Zig-Zag.</p>
</td></tr>
<tr><td><code id="can_stick">can_stick</code></td>
<td>
<p>Optional logical vector indicating which non-intercept
population-level coefficients are candidates for selection. Length
must match the number of supported coefficients. If omitted, all
non-intercept population-level coefficients are candidates.</p>
</td></tr>
<tr><td><code id="model_prior">model_prior</code></td>
<td>
<p>A [bernoulli()] or [betabernoulli()] object specifying
the prior on model space.</p>
</td></tr>
<tr><td><code id="kappa">kappa</code></td>
<td>
<p>Optional numeric vector of slab densities at zero for each
stickable coordinate (kappa in the sticky PDMP literature). If omitted,
derived automatically from the brms prior specification (only
'normal(0, s)' and 'student_t(df, 0, s)' are supported for automatic
derivation).</p>
</td></tr>
<tr><td><code id="slab_prior">slab_prior</code></td>
<td>
<p>Optional dependent slab prior created by
[dense_gaussian_slab()], [exchangeable_gaussian_slab()], or
[loglinear_gaussian_scale_slab()]. Mutually exclusive with 'kappa'. Full-data
dependent slabs use the brms prior-only data as the base prior target;
subsampled brms dependent slabs are not yet supported. Adaptive Boomerang
is supported with 'GridThinningStrategy' and requires positive warmup
(automatically set to one fifth of the sampling time when omitted).</p>
</td></tr>
<tr><td><code id="stanvars">stanvars</code></td>
<td>
<p>Optional 'stanvar' object for custom Stan code.</p>
</td></tr>
<tr><td><code id="sample_prior">sample_prior</code></td>
<td>
<p>Currently only '&quot;no&quot;' is supported.</p>
</td></tr>
<tr><td><code id="save_model">save_model</code></td>
<td>
<p>Optional file path to save the generated Stan code.</p>
</td></tr>
<tr><td><code id="control">control</code></td>
<td>
<p>Additional call-local settings from <code>sampler_control</code>.</p>
</td></tr>
<tr><td><code id="compile_options">compile_options</code></td>
<td>
<p>Settings from <code>stan_compile_options</code>.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Additional arguments passed to [brms::brm()] for model setup.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>PDMP samplers do not produce NUTS-style diagnostics. The 'lp__',
'accept_stat__', and related diagnostic columns are set to zero.
Functions relying on NUTS diagnostics (e.g., 'pairs()' divergence
plots, 'nuts_params()') will not produce meaningful output.
</p>
<p>'loo()' works because brms computes 'log_lik' directly from model
parameters, not from 'lp__'.
</p>
<p>With a single chain, 'Rhat' will report 'NA'. Use 'n_chains &gt;= 2'
for convergence diagnostics.
</p>
<p>Subsampling acceleration is enabled when the model has certified affine predictor,
likelihood, and dynamics providers. The first provider batch covers weighted
and subsetted Bernoulli/binomial logit models, categorical/multinomial logit,
Poisson log models (including 'rate()'), and Gaussian location-scale models
(including known 'se()' values and affine sigma predictors). Independent
affine multivariate Bernoulli responses are also supported. Unknown geometry
falls back to the exact full-gradient sampler. Priors, Jacobians, and
unconditional custom target additions remain in an opaque deterministic
BridgeStan provider.
</p>
<p>PDMPSamplers.jl applies 'N / m', draws a fresh subsampling subset per proposal,
uses dynamics-specific linear or harmonic trajectory bounds, and reuses the
accepted stochastic gradient for the event. Observation subsampling supports
'GridThinningStrategy' generally and 'ThinningStrategy' when the residual
envelope has an affine global roof (or the trajectory is periodic).
</p>


<h3>Value</h3>

<p>A 'brmsfit' object.
</p>
```

```@raw html
<span id="r-sampler_control"></span>
```

## `sampler_control`

```@raw html
<h3>Description</h3>

<p>Pass this object as <code>control</code> to <code>pdmp_sample()</code>,
<code>pdmp_sample_from_stanmodel()</code> or <code>brm_pdmp()</code>. Settings apply to that call
only, including when a worker is reused for another model.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>sampler_control(
  grid_n_min = NULL,
  grid_bound_violation = NULL,
  adaptive_boomerang_refresh_rate = 0.1,
  warmup_stop = NULL,
  profile = FALSE,
  research_diagnostics = FALSE,
  hcv_cache = TRUE,
  check_fast_deterministic = FALSE
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="grid_n_min">grid_n_min</code></td>
<td>
<p>Minimum grid size, or <code>NULL</code> for <code>min(grid_n, 5)</code>.</p>
</td></tr>
<tr><td><code id="grid_bound_violation">grid_bound_violation</code></td>
<td>
<p>Grid bound violation policy: <code>NULL</code> uses the
sampler default; otherwise <code>"throw"</code>, <code>"shrink"</code>, or <code>"count"</code>.</p>
</td></tr>
<tr><td><code id="adaptive_boomerang_refresh_rate">adaptive_boomerang_refresh_rate</code></td>
<td>
<p>Initial refresh rate for adaptive
Boomerang dynamics. The default is 0.1.</p>
</td></tr>
<tr><td><code id="warmup_stop">warmup_stop</code></td>
<td>
<p>Optional named list configuring adaptive warmup stopping:
<code>min_time</code>, <code>max_time</code>, <code>stable_time</code>, <code>min_events</code>, and <code>check_every</code>.
An empty list enables defaults (one quarter, all, and one quarter of
<code>t_warmup</code>, 100 events, and a check every 25 events); <code>NULL</code> disables it.
Supported by custom-Stan sampling.</p>
</td></tr>
<tr><td><code id="profile">profile</code></td>
<td>
<p>Collect bridge-phase timings for custom-Stan sampling.</p>
</td></tr>
<tr><td><code id="research_diagnostics">research_diagnostics</code></td>
<td>
<p>Collect additional OMRF allocation and HVP timing
diagnostics. These measurements add overhead.</p>
</td></tr>
<tr><td><code id="hcv_cache">hcv_cache</code></td>
<td>
<p>Reuse OMRF Hessian products along a flight.</p>
</td></tr>
<tr><td><code id="check_fast_deterministic">check_fast_deterministic</code></td>
<td>
<p>Compare the fast OMRF deterministic gradient
with its generic evaluation. Intended for diagnostics; adds overhead.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A sampler control list.
</p>
```

```@raw html
<span id="r-stan_compile_options"></span>
```

## `stan_compile_options`

```@raw html
<h3>Description</h3>

<p>Pass as <code>compile_options</code> to <code>pdmp_sample_from_stanmodel()</code>, <code>brm_pdmp()</code>
or <code>compile_pdmp_stan_model()</code>. Existing compiled libraries can be reused;
use a separate cache directory when changing toolchains.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>stan_compile_options(
  cache = FALSE,
  cache_dir = NULL,
  native = FALSE,
  optimize = FALSE
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="cache">cache</code></td>
<td>
<p>Use a content-addressed compilation cache.</p>
</td></tr>
<tr><td><code id="cache_dir">cache_dir</code></td>
<td>
<p>Cache directory, or <code>NULL</code> for the Julia depot's
<code>pdmpsamplersr/bridgestan</code> directory.</p>
</td></tr>
<tr><td><code id="native">native</code></td>
<td>
<p>Enable native CPU compiler instructions.</p>
</td></tr>
<tr><td><code id="optimize">optimize</code></td>
<td>
<p>Enable Stan's <code>--O1</code> optimization.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A compilation options list.
</p>
```

```@raw html
<span id="r-support_boundary_control"></span>
```

## `support_boundary_control`

```@raw html
<h3>Description</h3>

<p>Build a control list for support-boundary handling in PDMP samplers.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>support_boundary_control(
  mode = c("error", "line_search", "line_search_truncated_refresh"),
  max_bisection_steps = 60L,
  time_rtol = 1e-08,
  time_atol = 1e-10,
  clip_fraction = 1 - 1e-10,
  max_refresh_attempts = 20L,
  refresh_probe_time = 1e-04,
  min_safe_time = 1e-12
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="mode">mode</code></td>
<td>
<p>Character string, how to handle support-boundary violations where
the target or gradient becomes undefined during forward trajectory probing.
One of <code>"error"</code> (default, fail fast), <code>"line_search"</code>
(localize the first invalid time via bisection, then error), or
<code>"line_search_truncated_refresh"</code> (heuristic BPS-family recovery that
first searches for ordinary events before the localized boundary, handles
such an event if one occurs first, and otherwise refreshes velocity at a
valid interior point).</p>
</td></tr>
<tr><td><code id="max_bisection_steps">max_bisection_steps</code></td>
<td>
<p>Integer, maximum bisection iterations.</p>
</td></tr>
<tr><td><code id="time_rtol">time_rtol</code></td>
<td>
<p>Numeric, relative tolerance for bisection.</p>
</td></tr>
<tr><td><code id="time_atol">time_atol</code></td>
<td>
<p>Numeric, absolute tolerance for bisection.</p>
</td></tr>
<tr><td><code id="clip_fraction">clip_fraction</code></td>
<td>
<p>Numeric in <code>(0, 1]</code>, fraction of the last-valid time
used as a safe interior point after localization. The truncated-refresh mode
may apply an additional conservative cap before refreshing.</p>
</td></tr>
<tr><td><code id="max_refresh_attempts">max_refresh_attempts</code></td>
<td>
<p>Integer, maximum number of refreshed velocities to
try if <code>mode = "line_search_truncated_refresh"</code> reaches the support
boundary before an ordinary event.</p>
</td></tr>
<tr><td><code id="refresh_probe_time">refresh_probe_time</code></td>
<td>
<p>Numeric, short forward probe time used to reject
immediately invalid refreshed velocities. A value of zero disables the
probe; otherwise the Julia fallback may use a slightly longer scale-aware
probe.</p>
</td></tr>
<tr><td><code id="min_safe_time">min_safe_time</code></td>
<td>
<p>Numeric, minimum time gap used when clipping away from the
localized boundary.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>Both <code>"line_search"</code> and <code>"line_search_truncated_refresh"</code> probe
along the linear ray <code class="reqn">x_0 + t v</code> and are therefore only valid for
BPS/ZigZag-family flows with linear dynamics. For non-linear flows (e.g.,
Boomerang) these modes fall back to <code>"error"</code> behavior.
</p>


<h3>Value</h3>

<p>A list suitable for the <code>support_boundary</code> argument.
</p>
```

```@raw html
<span id="r-online_summary_options"></span>
```

## `online_summary_options`

```@raw html
<h3>Description</h3>

<p>Configures exact continuous-time summaries that are accumulated while a
streaming chain runs. Instead of storing every event, the sampler keeps,
for 'batches' equal physical-time batches of the retained phase, the
integrals of <code class="reqn">x</code>, <code class="reqn">x^2</code>, <code class="reqn">x^4</code> and of the free (non-zero)
indicator for every coordinate, the integrals of the number of free
coordinates and its square, and the number of sticky transitions per
coordinate. Optionally it also records the state on a regular time grid.
Memory and disk use therefore do not grow with the number of events.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>online_summary_options(
  batches = 400L,
  grid_spacing = NA_real_,
  write_events = FALSE,
  warmup_batches = 0L
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="batches">batches</code></td>
<td>
<p>Number of equal physical-time batches of the retained phase.</p>
</td></tr>
<tr><td><code id="grid_spacing">grid_spacing</code></td>
<td>
<p>Physical time between state snapshots, or 'NA' for none.</p>
</td></tr>
<tr><td><code id="write_events">write_events</code></td>
<td>
<p>If 'FALSE' (default), no retained-phase events are
written; the online summaries are the only retained output.</p>
</td></tr>
<tr><td><code id="warmup_batches">warmup_batches</code></td>
<td>
<p>Number of equal physical-time batches of the warmup
phase with the same integrals (no snapshots), or 0 (default) for none.
They are read with 'read_online_summaries(path, phase = &quot;warmup&quot;)'.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>Pass the result as 'online_summaries' to [pdmp_sample()] or
[pdmp_sample_from_stanmodel()] together with 'trace_storage = &quot;streaming&quot;',
and read the summaries afterwards with [read_online_summaries()].
</p>


<h3>Value</h3>

<p>A list of class 'pdmp_online_summaries'.
</p>
```

