# Results

[R function reference](../reference.md)

Generated from the PDMPSamplersR help files. The R package is the source of truth.

```@raw html
<span id="r-mean.pdmp_result"></span>
```

## `mean.pdmp_result`

```@raw html
<h3>Description</h3>

<p>Computes the time-weighted mean from the continuous occupation measure.
When <code>transforms</code> is supplied, returns the mean of constrained
(transformed) parameters.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>## S3 method for class 'pdmp_result'
mean(x, transforms = NULL, chain = NULL, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A <code>pdmp_result</code> object.</p>
</td></tr>
<tr><td><code id="transforms">transforms</code></td>
<td>
<p>Optional list of transform specifications (see
<code>identity_transform</code>).</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to use, or <code>NULL</code> to pool all
chains (default: <code>NULL</code>).</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Ignored.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>With several chains and <code>chain = NULL</code>, all estimators pool the chains
as one process observed for their combined time: each chain is weighted by
the length of its run. PDMPSamplers.jl computes the pooled values, so they
are identical in R and Julia.
</p>


<h3>Value</h3>

<p>Numeric vector of length <code>x$d</code>.
</p>
```

```@raw html
<span id="r-median.pdmp_result"></span>
```

## `median.pdmp_result`

```@raw html
<h3>Description</h3>

<p>Continuous-time median of PDMP trace
</p>


<h3>Usage</h3>

<pre><code class='language-R'>## S3 method for class 'pdmp_result'
median(
  x,
  na.rm = FALSE,
  transforms = NULL,
  chain = NULL,
  coordinate = -1L,
  ...
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A <code>pdmp_result</code> object.</p>
</td></tr>
<tr><td><code id="na.rm">na.rm</code></td>
<td>
<p>Ignored (present for compatibility with the median).</p>
</td></tr>
<tr><td><code id="transforms">transforms</code></td>
<td>
<p>Optional list of transform specifications.</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to use, or <code>NULL</code> to pool all
chains (default: <code>NULL</code>).</p>
</td></tr>
<tr><td><code id="coordinate">coordinate</code></td>
<td>
<p>Integer coordinate index, or -1 for all (default: -1).</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Ignored.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Numeric vector or scalar.
</p>
```

```@raw html
<span id="r-var"></span>
```

## `var`

```@raw html
<h3>Description</h3>

<p>Generic function for variance computation. Falls back to
<code>var</code> for non-PDMP objects.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>var(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>Object.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to methods.</p>
</td></tr>
</table>
```

```@raw html
<span id="r-sd"></span>
```

## `sd`

```@raw html
<h3>Description</h3>

<p>Standard deviation
</p>


<h3>Usage</h3>

<pre><code class='language-R'>sd(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>Object.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to methods.</p>
</td></tr>
</table>
```

```@raw html
<span id="r-cov"></span>
```

## `cov`

```@raw html
<h3>Description</h3>

<p>Covariance matrix
</p>


<h3>Usage</h3>

<pre><code class='language-R'>cov(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>Object.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to methods.</p>
</td></tr>
</table>
```

```@raw html
<span id="r-cor"></span>
```

## `cor`

```@raw html
<h3>Description</h3>

<p>Correlation matrix
</p>


<h3>Usage</h3>

<pre><code class='language-R'>cor(x, ...)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>Object.</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Passed to methods.</p>
</td></tr>
</table>
```

```@raw html
<span id="r-quantile.pdmp_result"></span>
```

## `quantile.pdmp_result`

```@raw html
<h3>Description</h3>

<p>When <code>chain = NULL</code>, the quantiles are those of the pooled
distribution of all chains (see <code>mean.pdmp_result</code>).
</p>


<h3>Usage</h3>

<pre><code class='language-R'>## S3 method for class 'pdmp_result'
quantile(
  x,
  probs,
  transforms = NULL,
  chain = NULL,
  coordinate = -1L,
  gc_interval = 16L,
  ...
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A <code>pdmp_result</code> object.</p>
</td></tr>
<tr><td><code id="probs">probs</code></td>
<td>
<p>Numeric vector of probabilities in (0, 1).</p>
</td></tr>
<tr><td><code id="transforms">transforms</code></td>
<td>
<p>Optional list of transform specifications.</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to use, or <code>NULL</code> to pool all
chains (default: <code>NULL</code>).</p>
</td></tr>
<tr><td><code id="coordinate">coordinate</code></td>
<td>
<p>Integer coordinate index, or -1 for all (default: -1).</p>
</td></tr>
<tr><td><code id="gc_interval">gc_interval</code></td>
<td>
<p>Collect temporary Julia workspaces after this many
coordinates when computing multiple untransformed quantiles (default 16).</p>
</td></tr>
<tr><td><code id="...">...</code></td>
<td>
<p>Ignored.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Numeric matrix (<code>length(probs)</code> x d) when <code>coordinate = -1</code>,
or a numeric vector of length <code>length(probs)</code> for a single coordinate.
</p>
```

```@raw html
<span id="r-ess"></span>
```

## `ess`

```@raw html
<h3>Description</h3>

<p>Estimate ESS per coordinate from a PDMP trace using the batch-means
method, without discretization.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>ess(x, ...)

## S3 method for class 'pdmp_result'
ess(x, chain = NULL, n_batches = 0L, ...)
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
chains, which sums the per-chain estimates (default: <code>NULL</code>).</p>
</td></tr>
<tr><td><code id="n_batches">n_batches</code></td>
<td>
<p>Integer, number of batches. 0 uses the default (default: 0).</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Numeric vector of length <code>x$d</code>.
</p>
```

```@raw html
<span id="r-cdf"></span>
```

## `cdf`

```@raw html
<h3>Description</h3>

<p>Compute the fraction of trajectory time with the transform of
coordinate <code>coordinate</code> at or below <code>q</code>.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>cdf(x, ...)

## S3 method for class 'pdmp_result'
cdf(x, q, coordinate, transforms = NULL, chain = NULL, ...)
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
<tr><td><code id="q">q</code></td>
<td>
<p>Numeric threshold.</p>
</td></tr>
<tr><td><code id="coordinate">coordinate</code></td>
<td>
<p>Integer coordinate index.</p>
</td></tr>
<tr><td><code id="transforms">transforms</code></td>
<td>
<p>Optional list of transform specifications.</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to use, or <code>NULL</code> to pool all
chains (default: <code>NULL</code>).</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Numeric scalar in [0, 1].
</p>
```

```@raw html
<span id="r-discretize"></span>
```

## `discretize`

```@raw html
<h3>Description</h3>

<p>Convert the continuous-time trace to a matrix of equally-spaced samples.
When <code>dt = NULL</code> (default), the number of discretization points is set
to the continuous-time ESS, preserving the full information content of the
trace.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>discretize(x, ...)

## S3 method for class 'pdmp_result'
discretize(x, dt = NULL, chain = NULL, ...)
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
<tr><td><code id="dt">dt</code></td>
<td>
<p>Numeric time step, or NULL to use adaptive discretization based on
continuous-time ESS (default: NULL).</p>
</td></tr>
<tr><td><code id="chain">chain</code></td>
<td>
<p>Integer, which chain to discretize. A discretization is a
single path, so it is required when there are several chains; with one
chain, <code>NULL</code> (the default) uses it.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Numeric matrix (samples x dimensions).
</p>
```

```@raw html
<span id="r-materialize"></span>
```

## `materialize`

```@raw html
<h3>Description</h3>

<p>Copies the chain skeleton (event times, positions, velocities) from Julia
into R and drops the live Julia reference. The returned object can be saved
with <code>saveRDS()</code> and all estimators continue to work after reloading,
provided Julia is set up before the first estimator call.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>materialize(x)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="x">x</code></td>
<td>
<p>A <code>pdmp_result</code>.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>This only needs to be called manually when the result was created with
<code>materialize = FALSE</code>; all sampling functions materialise the trace
automatically by default.
</p>


<h3>Value</h3>

<p>A <code>pdmp_result</code> with <code>$chains = NULL</code> and
<code>$skeleton</code> populated.
</p>
```

```@raw html
<span id="r-read_online_summaries"></span>
```

## `read_online_summaries`

```@raw html
<h3>Description</h3>

<p>Read online summaries written by a streaming chain
</p>


<h3>Usage</h3>

<pre><code class='language-R'>read_online_summaries(path, phase = c("main", "warmup"))
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="path">path</code></td>
<td>
<p>Either the 'online_summaries.bin' file or a streaming trace
directory, which is searched for exactly one such file of 'phase'.</p>
</td></tr>
<tr><td><code id="phase">phase</code></td>
<td>
<p>'&quot;main&quot;' (default) for the retained phase or '&quot;warmup&quot;';
only used when 'path' is a directory.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A list with the batch integrals ('x', 'x2', 'x4', 'free': one column
per batch), 'batch_time', 'size', 'size2' (integrals of the number of free
coordinates and its square), 'transitions' (per coordinate and batch),
the snapshot grid ('grid_times', 'grid_x', 'grid_free'), and
'start_time', 'width' and 'end_time' of the retained phase.
</p>
```

