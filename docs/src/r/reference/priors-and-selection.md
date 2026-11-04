# Priors and selection

[R function reference](../reference.md)

Generated from the PDMPSamplersR help files. The R package is the source of truth.

```@raw html
<span id="r-bernoulli"></span>
```

## `bernoulli`

```@raw html
<h3>Description</h3>

<p>Specifies independent Bernoulli inclusion probabilities for use as a
model prior in spike-and-slab sampling.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>bernoulli(prob = 0.5)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="prob">prob</code></td>
<td>
<p>Numeric value or vector of inclusion probabilities, each
between 0 and 1 (default: 0.5). A single value is recycled across
all dimensions.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>An object of class <code>"bernoulli"</code>.
</p>


<h3>See Also</h3>

<p><code>betabernoulli</code> for a Beta-Bernoulli alternative.
</p>
```

```@raw html
<span id="r-betabernoulli"></span>
```

## `betabernoulli`

```@raw html
<h3>Description</h3>

<p>Specifies a Beta-Bernoulli model prior for spike-and-slab sampling.
The inclusion probability is given a <code class="reqn">\text{Beta}(a, b)</code>
prior, which induces a multiplicity correction.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>betabernoulli(a = 1, b = 1)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="a">a</code></td>
<td>
<p>Positive numeric, first shape parameter of the Beta distribution (default: 1).</p>
</td></tr>
<tr><td><code id="b">b</code></td>
<td>
<p>Positive numeric, second shape parameter of the Beta distribution (default: 1).</p>
</td></tr>
</table>


<h3>Value</h3>

<p>An object of class <code>"beta-bernoulli"</code>.
</p>


<h3>See Also</h3>

<p><code>bernoulli</code> for fixed inclusion probabilities.
</p>
```

```@raw html
<span id="r-dependent_slab_priors"></span>
```

## `dependent_slab_priors`

Also documents: `dense_gaussian_slab`, `exchangeable_gaussian_slab`, `loglinear_gaussian_scale_slab`.

```@raw html
<h3>Description</h3>

<p>These constructors describe slab priors for the dependent aggregate sticky
Julia path. They are intentionally distinct from the legacy <code>kappa</code> /
<code>parameter_prior</code> sticky interface: <code>slab_prior</code> uses the new
boundary-density convention in which outgoing boundary velocities are drawn
by the aggregate sticky kernel.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>dense_gaussian_slab(mean, cov, coef = NULL)

exchangeable_gaussian_slab(mean = 0, u, v, coef = NULL)

loglinear_gaussian_scale_slab(
  log_base_sd,
  logscale,
  logscale_design,
  coef = NULL
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="mean">mean</code></td>
<td>
<p>Common mean for an exchangeable Gaussian slab.</p>
</td></tr>
<tr><td><code id="cov">cov</code></td>
<td>
<p>Positive-definite covariance matrix for a fixed Gaussian slab.</p>
</td></tr>
<tr><td><code id="coef">coef</code></td>
<td>
<p>Optional character vector of unconstrained coefficient names, or
integer vector of 1-based unconstrained parameter indices, controlled by
this slab. If omitted, the bridge uses <code>can_stick</code>.</p>
</td></tr>
<tr><td><code id="u">u</code></td>
<td>
<p>Diagonal covariance component in <code class="reqn">u I + v 11'</code>.</p>
</td></tr>
<tr><td><code id="v">v</code></td>
<td>
<p>Rank-one covariance component in <code class="reqn">u I + v 11'</code>.</p>
</td></tr>
<tr><td><code id="log_base_sd">log_base_sd</code></td>
<td>
<p>Numeric log base standard deviations. A scalar is
recycled over the selected coefficients.</p>
</td></tr>
<tr><td><code id="logscale">logscale</code></td>
<td>
<p>Integer or character vector of unconstrained log-scale
coordinate indices/names.</p>
</td></tr>
<tr><td><code id="logscale_design">logscale_design</code></td>
<td>
<p>Numeric coefficient-by-logscale design matrix. Sparse
matrices are accepted and retained as a compact specification.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A slab prior specification for the dependent aggregate sticky
bridge.
</p>
```

```@raw html
<span id="r-inclusion_probs"></span>
```

## `inclusion_probs`

```@raw html
<h3>Description</h3>

<p>For spike-and-slab models, compute the fraction of time each
coordinate spent away from zero.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>inclusion_probs(x, ...)

## S3 method for class 'pdmp_result'
inclusion_probs(x, chain = NULL, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A <code>pdmp_result</code> object.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to methods.</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to use, or <code>NULL</code> to pool all
chains (default: <code>NULL</code>).</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Numeric vector of length <code>x$d</code>.
</p>
```

```@raw html
<span id="r-inclusion_prob"></span>
```

## `inclusion_prob`

```@raw html
<h3>Description</h3>

<p>Extracts per-coefficient inclusion probabilities from a sticky
'brmsfit' fitted with [brm_pdmp()] and 'sticky = TRUE'. Each
probability estimates <code class="reqn">P(\gamma_k = 1 \mid y)</code>, the posterior
probability that coefficient <code class="reqn">\beta_k</code> is non-zero under the
user-specified spike-and-slab prior.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>inclusion_prob(x, ...)

## Default S3 method:
inclusion_prob(x, ...)

## S3 method for class 'brmsfit'
inclusion_prob(x, chain = NULL, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A 'brmsfit' fitted with 'sticky = TRUE'.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Ignored.</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to extract, or 'NULL' to pool all
chains (default: 'NULL'), weighting each chain by the length of its run.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A named numeric vector in <code class="reqn">[0, 1]</code>, one entry per
stickable population-level coefficient.
</p>
```

```@raw html
<span id="r-inclusion_bf"></span>
```

## `inclusion_bf`

```@raw html
<h3>Description</h3>

<p>Computes the inclusion Bayes factor for each stickable coefficient:
</p>
<p style="text-align: center;"><code class="reqn">BF_{\text{incl},k} = \frac{P(\gamma_k = 1 \mid y)}{P(\gamma_k = 0 \mid y)}
      \cdot \frac{P(\gamma_k = 0)}{P(\gamma_k = 1)}</code>
</p>



<h3>Usage</h3>

<pre><code class='language-R'>inclusion_bf(x, ...)

## Default S3 method:
inclusion_bf(x, ...)

## S3 method for class 'brmsfit'
inclusion_bf(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A 'brmsfit' fitted with 'sticky = TRUE' using a version of
[brm_pdmp()] that stores the model prior in the sticky metadata.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to [inclusion_prob()].</p>
</td></tr>
</table>


<h3>Details</h3>

<p>This answers &quot;how much did the data update the prior odds in favour of
inclusion?&quot; Values greater than 1 indicate evidence for inclusion; values
less than 1 indicate evidence against. Unlike inclusion probabilities alone,
the Bayes factor separates the likelihood signal from the prior.
</p>


<h3>Value</h3>

<p>A named numeric vector of inclusion Bayes factors, one per
stickable coefficient.
</p>
```

```@raw html
<span id="r-model_averaged_mean"></span>
```

## `model_averaged_mean`

```@raw html
<h3>Description</h3>

<p>Returns <code class="reqn">E[\beta_k \mid y]</code> for each stickable population-level
coefficient by averaging the posterior draws produced by the sticky PDMP
trajectory. Because the trajectory spends time at zero (the spike) in
proportion to the posterior spike mass, this equals the spike-and-slab
model-averaged mean provided the sampler has converged and the zero-dwell
time is correctly attributed. Use [inclusion_prob()] to inspect the
corresponding inclusion probabilities.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>model_averaged_mean(x, ...)

## S3 method for class 'brmsfit'
model_averaged_mean(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A 'brmsfit' fitted with 'sticky = TRUE'.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Ignored.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A named numeric vector of posterior means.
</p>
```

```@raw html
<span id="r-median_probability_model"></span>
```

## `median_probability_model`

```@raw html
<h3>Description</h3>

<p>Returns the names of population-level coefficients with posterior
inclusion probability above 'threshold' (default 0.5). This is the
*median probability model*: the model that includes each variable for
which the posterior probability of inclusion exceeds one half.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>median_probability_model(x, ...)

## S3 method for class 'brmsfit'
median_probability_model(x, threshold = 0.5, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A 'brmsfit' fitted with 'sticky = TRUE'.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to [inclusion_prob()].</p>
</td></tr>
<tr><td><code id="threshold">threshold</code></td>
<td>
<p>Numeric scalar in <code class="reqn">[0, 1]</code> (default: 0.5).</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A character vector of coefficient names.
</p>


<h3>Collinearity caveat</h3>

<p>The median probability model can behave poorly under moderate or
strong collinearity. Coefficient-wise inclusion and term-wise
predictive reduction are not the same task; see the package
documentation for guidance.
</p>
```

```@raw html
<span id="r-check_inclusion_stability"></span>
```

## `check_inclusion_stability`

```@raw html
<h3>Description</h3>

<p>Compares inclusion probability estimates across chains to help assess
convergence of the sticky sampler. Returns a data frame with cross-chain
summary statistics for each stickable coefficient.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>check_inclusion_stability(x, ...)

## Default S3 method:
check_inclusion_stability(x, ...)

## S3 method for class 'brmsfit'
check_inclusion_stability(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A 'brmsfit' fitted with 'sticky = TRUE'.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Ignored.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A data frame with columns 'coefficient', 'mean', 'sd', 'min',
'max', and 'range' summarising cross-chain variability.
</p>
```

```@raw html
<span id="r-plot_inclusion"></span>
```

## `plot_inclusion`

```@raw html
<h3>Description</h3>

<p>Produces a horizontal bar chart of coefficient-wise inclusion
probabilities from a sticky 'brmsfit', with a vertical reference
line at 'threshold'.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>plot_inclusion(x, ...)

## S3 method for class 'brmsfit'
plot_inclusion(x, threshold = 0.5, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A 'brmsfit' fitted with 'sticky = TRUE'.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to [inclusion_prob()].</p>
</td></tr>
<tr><td><code id="threshold">threshold</code></td>
<td>
<p>Numeric scalar in <code class="reqn">[0, 1]</code>, the reference line
position (default: 0.5).</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A 'ggplot' object.
</p>
```

