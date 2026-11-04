# Stan and subsampling

[R function reference](../reference.md)

Generated from the PDMPSamplersR help files. The R package is the source of truth.

```@raw html
<span id="r-stan_subsampling"></span>
```

## `stan_subsampling`

Also documents: `pdmp_subsample_hpp_path`, `compile_pdmp_stan_model`, `stan_residual_envelope`, `omrf_residual_envelope`, `stan_parameter_mapping`.

```@raw html
<h3>Description</h3>

<p>'pdmp_subsample_hpp_path()' returns the installed thread-local subset-hook
header. 'compile_pdmp_stan_model()' compiles a Stan source with the required
'&ndash;allow-undefined' and 'USER_HEADER' settings.
</p>
<p>Specifies exact subsampling for an ordinal Markov random field (OMRF)
likelihood with one likelihood contribution per person. The residual of a
person's gradient is taken relative to an analytic, undamped Hessian control
variate at the anchor, and every grid cell gets a certified roof on the
summed residual rates.
Threshold and interaction arguments are Stan unconstrained block names and
are resolved at sampler initialization. The prior gradient is analytic, so
'prior_standata' passed to [stan_subsampling()] must follow one of the
package's OMRF prior contracts.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>pdmp_subsample_hpp_path()

compile_pdmp_stan_model(path, compile_options = stan_compile_options())

stan_residual_envelope(weights, growth_rates = NULL)

stan_subsampling(
  n_observations,
  subsample_size,
  prior_standata,
  residual_envelope,
  anchor = NULL
)

omrf_residual_envelope(
  X,
  seen,
  thresholds,
  interactions,
  exact_selected_refinement = TRUE,
  person_signed_roof = FALSE,
  signed_only_person_roof = FALSE,
  joint_signed_group_size = 0L,
  shared_current_categorical = FALSE
)

stan_parameter_mapping(path_to_stanmodel, standata, parameters = NULL)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="path">path</code></td>
<td>
<p>Path to a Stan source file.</p>
</td></tr>
<tr><td><code id="compile_options">compile_options</code></td>
<td>
<p>Settings from <code>stan_compile_options</code>.</p>
</td></tr>
<tr><td><code id="weights">weights</code></td>
<td>
<p>Nonnegative component-by-observation curvature weights.</p>
</td></tr>
<tr><td><code id="growth_rates">growth_rates</code></td>
<td>
<p>Optional nonnegative growth coefficient per component.</p>
</td></tr>
<tr><td><code id="n_observations">n_observations</code></td>
<td>
<p>Total number of subsampling likelihood contributions.</p>
</td></tr>
<tr><td><code id="subsample_size">subsample_size</code></td>
<td>
<p>Number drawn without replacement at each proposal.</p>
</td></tr>
<tr><td><code id="prior_standata">prior_standata</code></td>
<td>
<p>Prior-only data list or JSON path for the same model.</p>
</td></tr>
<tr><td><code id="residual_envelope">residual_envelope</code></td>
<td>
<p>A certified object from [stan_residual_envelope()]
or a recognized package provider such as [omrf_residual_envelope()].</p>
</td></tr>
<tr><td><code id="anchor">anchor</code></td>
<td>
<p>Optional unconstrained anchor.</p>
</td></tr>
<tr><td><code id="X">X</code></td>
<td>
<p>Integer person-by-node matrix encoded from zero.</p>
</td></tr>
<tr><td><code id="seen">seen</code></td>
<td>
<p>Number of observed categories per node.</p>
</td></tr>
<tr><td><code id="thresholds">thresholds</code></td>
<td>
<p>Threshold parameter block name.</p>
</td></tr>
<tr><td><code id="interactions">interactions</code></td>
<td>
<p>Interaction parameter block name; edge order is the
upper triangle '(1,2), (1,3), ..., (P-1,P)'.</p>
</td></tr>
<tr><td><code id="exact_selected_refinement">exact_selected_refinement</code></td>
<td>
<p>If &lsquo;TRUE' (default), the selected person&rsquo;s
residual is evaluated once while thinning against the subset bound and
reused for the candidate rate. 'FALSE' thins against the drawn roof only.</p>
</td></tr>
<tr><td><code id="person_signed_roof">person_signed_roof</code></td>
<td>
<p>Linear BPS certificate policy. Signed node
residual endpoints are summed within each person before taking an
absolute bound; the factor-summed certificate remains a valid fallback.
The estimator and retained target are unchanged. Other flows keep the
ordinary roof.</p>
</td></tr>
<tr><td><code id="signed_only_person_roof">signed_only_person_roof</code></td>
<td>
<p>Linear BPS policy which builds the certified
signed whole-person endpoint roof without also building the factor-summed
fallback table. It preserves the estimator and uses pointwise factor
refinement for the selected person. Requires 'person_signed_roof = TRUE'.</p>
</td></tr>
<tr><td><code id="joint_signed_group_size">joint_signed_group_size</code></td>
<td>
<p>Linear BPS joint signed marked roof. Use '4'
for fixed seeded groups of four, '16', '32' or '64' for weighted ragged
groups, or '0' to keep the ordinary scheduler. Requires
'signed_only_person_roof = TRUE'.</p>
</td></tr>
<tr><td><code id="shared_current_categorical">shared_current_categorical</code></td>
<td>
<p>Evaluator that reuses current categorical
moments for exact active-neighbour patterns. Requires joint groups.</p>
</td></tr>
<tr><td><code id="path_to_stanmodel">path_to_stanmodel</code></td>
<td>
<p>Stan source or compiled-library path.</p>
</td></tr>
<tr><td><code id="standata">standata</code></td>
<td>
<p>Named data list or JSON path.</p>
</td></tr>
<tr><td><code id="parameters">parameters</code></td>
<td>
<p>Optional unconstrained names, block prefixes, or indices.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>The header path, or the compiled shared-library path.
</p>
<p>A declarative residual-envelope specification.
</p>
<p>A validated observation-subsampling specification.
</p>
<p>A certified Stan residual-envelope specification.
</p>
<p>A data frame with resolved 1-based indices and names.
</p>
```

```@raw html
<span id="r-stan_subsampling_diagnostics"></span>
```

## `stan_subsampling_diagnostics`

```@raw html
<h3>Description</h3>

<p>Evaluates the persistent full, prior-only, and selected Stan modes at one
position without starting a sampler. The selected likelihood gradient is
unscaled; 'selected_scale' reports the 'N / m' factor applied by Julia when
forming the subsampling gradient. The residual event-rate diagnostic uses the
same typed envelope and trajectory geometry as sampling.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>stan_subsampling_diagnostics(
  path_to_stanmodel,
  standata,
  subsampling,
  position,
  subset,
  velocity = NULL,
  flow = c("ZigZag", "BouncyParticle", "AdaptiveBoomerang"),
  flow_mean = NULL,
  flow_cov = NULL,
  phase = c("main", "warmup")
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="path_to_stanmodel">path_to_stanmodel</code></td>
<td>
<p>Stan source or compiled-library path containing the
package subset hook.</p>
</td></tr>
<tr><td><code id="standata">standata</code></td>
<td>
<p>Full-likelihood Stan data list or JSON path.</p>
</td></tr>
<tr><td><code id="subsampling">subsampling</code></td>
<td>
<p>A specification from [stan_subsampling()].</p>
</td></tr>
<tr><td><code id="position">position</code></td>
<td>
<p>Finite unconstrained parameter vector.</p>
</td></tr>
<tr><td><code id="subset">subset</code></td>
<td>
<p>One-based observation indices, with exactly the configured
subsample size and no duplicates.</p>
</td></tr>
<tr><td><code id="velocity">velocity</code></td>
<td>
<p>Finite diagnostic velocity. Defaults to a vector of ones.</p>
</td></tr>
<tr><td><code id="flow">flow</code></td>
<td>
<p>Supported subsampling flow used for the rate diagnostic.</p>
</td></tr>
<tr><td><code id="flow_mean">flow_mean</code></td>
<td>
<p>Optional flow mean.</p>
</td></tr>
<tr><td><code id="flow_cov">flow_cov</code></td>
<td>
<p>Optional flow covariance.</p>
</td></tr>
<tr><td><code id="phase">phase</code></td>
<td>
<p>Diagnostic sampling phase. For staged HCV specifications,
'&quot;main&quot;' activates HCV and '&quot;warmup&quot;' uses the first-order fallback.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A list of gradient-closure, residual-rate, envelope, and construction
diagnostics.
</p>
```

```@raw html
<span id="r-position_anchor_bank"></span>
```

## `position_anchor_bank`

```@raw html
<h3>Description</h3>

<p>Builds the anchor bank used by position-selected OMRF subsampling from
states of a previous (typically full-gradient) chain, for example the online
grid snapshots of [read_online_summaries()]. It applies the same rule the
sampler uses for its own warmup: coordinate scales are the standard
deviations of the states (floored at a tenth of their median), the first
anchor is the last state, and further anchors are added by greedy
farthest-point selection in the scaled metric.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>position_anchor_bank(
  standata,
  subsampling,
  parameter_names,
  states,
  capacity = 8L,
  metric = c("scaled", "hessian")
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="standata">standata</code></td>
<td>
<p>The Stan data of the subsampled fit.</p>
</td></tr>
<tr><td><code id="subsampling">subsampling</code></td>
<td>
<p>The [stan_subsampling()] object of the subsampled fit.</p>
</td></tr>
<tr><td><code id="parameter_names">parameter_names</code></td>
<td>
<p>Unconstrained parameter names, in sampler order.</p>
</td></tr>
<tr><td><code id="states">states</code></td>
<td>
<p>Numeric matrix with one state per column.</p>
</td></tr>
<tr><td><code id="capacity">capacity</code></td>
<td>
<p>Maximum number of anchors.</p>
</td></tr>
<tr><td><code id="metric">metric</code></td>
<td>
<p>Anchor selection metric, '&quot;scaled&quot;' or '&quot;hessian&quot;'.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>During sampling each state uses the anchor nearest in a shared metric, so
the anchor is a function of position and the Voronoi cells are bounded by
hyperplanes. 'metric = &quot;scaled&quot;' uses the coordinate scales above;
'metric = &quot;hessian&quot;' uses the mean likelihood Hessian of the anchors plus
the identity, which approximates the posterior precision and so predicts
the control-variate error better (it requires global HCV anchors).
</p>
<p>Pass the result as 'subsampling_position_anchor_bank_spec' to
[pdmp_sample_from_stanmodel()], with 'subsampling_anchor_selection_policy =
&quot;position&quot;', 't_warmup = 0' and a frozen flow.
</p>


<h3>Value</h3>

<p>A version-1 position anchor bank specification.
</p>
```

```@raw html
<span id="r-write_stan_json"></span>
```

## `write_stan_json`

```@raw html
<h3>Description</h3>

<p>Write data to a JSON file readable by Stan
</p>


<h3>Usage</h3>

<pre><code class='language-R'>write_stan_json(data, file, always_decimal = FALSE)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="data">data</code></td>
<td>
<p>(list) A named list of <span class="rlang"><b>R</b></span> objects.</p>
</td></tr>
<tr><td><code id="file">file</code></td>
<td>
<p>(string) The path to where the data file should be written.</p>
</td></tr>
<tr><td><code id="always_decimal">always_decimal</code></td>
<td>
<p>(logical) Force generate non-integers with decimal
points to better distinguish between integers and floating point values.
If 'TRUE' all <span class="rlang"><b>R</b></span> objects in 'data' intended for integers must be of integer
type.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>Note: this method is copied from the cmdstanr package to avoid a dependency
on a non-CRAN package. It's copied at commit [edccf2d2f6449e7d80626a3ee6cc93845e82915b](https://github.com/stan-dev/cmdstanr/blob/edccf2d2f6449e7d80626a3ee6cc93845e82915b/R/data.R#L59).
As such, this file and the code therein follows the same license as cmdstanr (BSD 3-Clause License),
see the LICENSE file of cmdstanr for details.
</p>
<p>'write_stan_json()' performs several conversions before writing the JSON
file:
</p>
<p>* 'logical' -&gt; 'integer' ('TRUE' -&gt; '1', 'FALSE' -&gt; '0')
* 'data.frame' -&gt; 'matrix' (via [data.matrix()])
* 'list' -&gt; 'array'
* 'table' -&gt; 'vector', 'matrix', or 'array' (depending on dimensions of table)
</p>
<p>The 'list' to 'array' conversion is intended to make it easier to prepare
the data for certain Stan declarations involving arrays:
</p>
<p>* 'vector[J] v[K]' (or equivalently 'array[K] vector[J] v ' as of Stan 2.27)
can be constructed in <span class="rlang"><b>R</b></span> as a list with 'K' elements where each element a
vector of length 'J'
* 'matrix[I,J] v[K]' (or equivalently 'array[K] matrix[I,J] m ' as of Stan
2.27 ) can be constructed in <span class="rlang"><b>R</b></span> as a list with 'K' elements where each element
an 'IxJ' matrix
</p>
<p>These can also be passed in from <span class="rlang"><b>R</b></span> as arrays instead of lists but the list
option is provided for convenience. Unfortunately for arrays with more than
one dimension, e.g., 'vector[J] v[K,L]' (or equivalently
'array[K,L] vector[J] v ' as of Stan 2.27) it is not possible to use an <span class="rlang"><b>R</b></span>
list and an array must be used instead. For this example the array in <span class="rlang"><b>R</b></span>
should have dimensions 'KxLxJ'.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>x &lt;- matrix(rnorm(10), 5, 2)
y &lt;- rpois(nrow(x), lambda = 10)
z &lt;- c(TRUE, FALSE)
data &lt;- list(N = nrow(x), K = ncol(x), x = x, y = y, z = z)

# write data to json file
file &lt;- tempfile(fileext = ".json")
write_stan_json(data, file)

# check the contents of the file
cat(readLines(file), sep = "\n")


# demonstrating list to array conversion
# suppose x is declared as `vector[3] x[2]` (or equivalently `array[2] vector[3] x`)
# we can use a list of length 2 where each element is a vector of length 3
data &lt;- list(x = list(1:3, 4:6))
file &lt;- tempfile(fileext = ".json")
write_stan_json(data, file)
cat(readLines(file), sep = "\n")

</code></pre>
```

