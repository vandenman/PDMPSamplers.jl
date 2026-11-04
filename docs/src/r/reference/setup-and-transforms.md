# Setup and transforms

[R function reference](../reference.md)

Generated from the PDMPSamplersR help files. The R package is the source of truth.

```@raw html
<span id="r-pdmpsamplers_setup"></span>
```

## `pdmpsamplers_setup`

```@raw html
<h3>Description</h3>

<p>Initializes the Julia runtime via <span class="pkg">JuliaCall</span> and sets up the
Julia project that PDMPSamplersR depends on. This installs all
required Julia packages and precompiles them. Call this once before
using any sampling functions.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>pdmpsamplers_setup(verbose = TRUE, project_dir = NULL)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="verbose">verbose</code></td>
<td>
<p>Logical, if <code>TRUE</code> (default), progress messages are
printed.</p>
</td></tr>
<tr><td><code id="project_dir">project_dir</code></td>
<td>
<p>Optional Julia project directory. Sets the R option
<code>pdmpsamplersr_config_julia_project_dir</code> for this R session; the same option
can be set before loading the package. Otherwise the user-cache project
is used. Configure each independent worker session separately.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Invisible <code>TRUE</code> if the setup succeeds.
</p>


<h3>See Also</h3>

<p><code>pdmpsamplers_update</code> to update Julia dependencies.
</p>
```

```@raw html
<span id="r-pdmpsamplers_status"></span>
```

## `pdmpsamplers_status`

```@raw html
<h3>Description</h3>

<p>Displays the current version of PDMPSamplers.jl and whether it is installed
from a local path or from GitHub.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>pdmpsamplers_status()
</code></pre>


<h3>Value</h3>

<p>Invisible <code>NULL</code>. Called for its side effects.
</p>


<h3>See Also</h3>

<p><code>pdmpsamplers_setup</code> to perform the initial setup,
<code>pdmpsamplers_update</code> to update Julia dependencies.
</p>
```

```@raw html
<span id="r-pdmpsamplers_update"></span>
```

## `pdmpsamplers_update`

```@raw html
<h3>Description</h3>

<p>Updates the Julia package dependencies used by PDMPSamplersR. If the
Julia project does not exist yet, it is created automatically.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>pdmpsamplers_update()
</code></pre>


<h3>Value</h3>

<p>Invisible <code>NULL</code>. Called for its side effects.
</p>


<h3>See Also</h3>

<p><code>pdmpsamplers_setup</code> to perform the initial setup.
</p>
```

```@raw html
<span id="r-use_local_pdmpsamplers"></span>
```

## `use_local_pdmpsamplers`

```@raw html
<h3>Description</h3>

<p>Switch the Julia project to use a local (development) copy of
PDMPSamplers.jl instead of the GitHub version, or switch back to the
GitHub version.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>use_local_pdmpsamplers(path = NULL)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="path">path</code></td>
<td>
<p>Character string giving the path to the local PDMPSamplers.jl
directory. If <code>NULL</code> (default), switches back to the GitHub version.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>Invisible <code>NULL</code>. Called for its side effects.
</p>
```

```@raw html
<span id="r-check_for_julia_setup"></span>
```

## `check_for_julia_setup`

```@raw html
<h3>Description</h3>

<p>Check if the Julia Project is Setup Properly
</p>


<h3>Usage</h3>

<pre><code class='language-R'>check_for_julia_setup(error_if_not_exists = FALSE, setup_if_not_exists = TRUE)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="error_if_not_exists">error_if_not_exists</code></td>
<td>
<p>logical, if TRUE, will throw an error if the Julia project is not found. If FALSE, will attempt to set up the project if setup_if_not_exists is TRUE.</p>
</td></tr>
<tr><td><code id="setup_if_not_exists">setup_if_not_exists</code></td>
<td>
<p>logical, if FALSE and 'error_if_not_exists' is FALSE, will not throw an error but printif the Julia project is not found. If TRUE, will attempt to set up the project if 'error_if_not_exists' is FALSE.</p>
</td></tr>
</table>


<h3>Details</h3>

<p>This function checks for the existence of the Julia project for PDMPSamplers.jl, and if it exists, loads it and the required packages. If it does not exist, it can either throw an error or attempt to set up the project, depending on the parameters. It is mostly useful for internal use to ensure the Julia environment is ready before calling any functions that depend on it, but can also be called directly by users if they want to check or set up the Julia project.
</p>


<h3>Value</h3>

<p>TRUE if the Julia project is successfully loaded or set up, <code>FALSE</code> otherwise (if 'error_if_not_exists' is also <code>FALSE</code>). If 'error_if_not_exists' is <code>TRUE</code>, will throw an error if the project cannot be loaded or set up.
</p>
```

```@raw html
<span id="r-cleanup_worker_state"></span>
```

## `cleanup_worker_state`

```@raw html
<h3>Description</h3>

<p>Clear package-owned Julia references left by sampling and request garbage
collection in both Julia and R. Use this after finishing a simulation task
in a persistent worker, before that worker starts another task. For example,
mirai daemons and foreach workers can process many models or data sets in
one R session; references held by JuliaCall can otherwise keep the previous
task's model and data alive after its R function returns.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>cleanup_worker_state()
</code></pre>


<h3>Details</h3>

<p>Call this <strong>inside the worker</strong>, not in the parent R session: it only cleans
the process in which it runs. Register it with <code>on.exit(..., add = TRUE)</code>
at the start of the worker's task function so cleanup also runs after an
error. Finish extracting summaries or saving the fit before cleanup. Remove
large local fit objects when they are no longer needed; objects still
referenced by your code cannot be collected.
</p>
<p>This function does not start or shut down Julia, unload the package or
compiled libraries, delete streaming files, or remove saved results. It
keeps the worker ready for another task. Allocators can retain freed memory,
so a successful cleanup does not guarantee an immediate fall in the process's
reported memory use. Cleanup is usually unnecessary when each task's worker
exits immediately, or during ordinary interactive sampling where you still
want to use the fit.
</p>
<p><strong>Persistent worker example.</strong>
The same task function can be submitted to mirai or foreach. Set the Julia
project in each worker when using a nondefault installation, and save a
materialized fit before returning a small result to the parent:
</p>
<div class="sourceCode r"><pre>run_condition &lt;- function(seed, path, project_dir = NULL) {
  on.exit(PDMPSamplersR::cleanup_worker_state(), add = TRUE)
  PDMPSamplersR::pdmpsamplers_setup(verbose = FALSE,
                                  project_dir = project_dir)
  fit &lt;- PDMPSamplersR::pdmp_sample(
    function(x) x, d = 2, T = 100, seed = seed,
    materialize = TRUE, show_progress = FALSE)
  saveRDS(fit, path)
  rm(fit)
  path
}

# After starting mirai daemons:
job &lt;- mirai::mirai(run_condition(seed, path),
  seed = 1L, path = tempfile(fileext = ".rds"),
  run_condition = run_condition)

# Or, after registering a foreach process-worker backend:
# library(foreach)
# foreach(i = seq_len(10), .export = "run_condition") %dopar% {
#   run_condition(i, sprintf("fit-%03d.rds", i))
# }
</pre></div>
<p>Workers must be able to write the supplied paths. Use shared storage if the
parent needs to read files produced on another machine. Export or pass
<code>run_condition</code> and its inputs according to your scheduler's conventions.
</p>


<h3>Value</h3>

<p>An invisible logical scalar: <code>TRUE</code> when the initialized bridge was
cleaned, <code>FALSE</code> otherwise. Attributes <code>initialized</code>, <code>reason</code> and <code>error</code>
distinguish <code>"cleaned"</code>, <code>"not_initialized"</code> (a harmless no-op), and
<code>"error"</code> (cleanup failed). An uninitialized worker is not initialized as
a side effect. Inspect these attributes when diagnosing worker cleanup;
cleanup errors are reported in the result rather than replacing an error
from the task being cleaned up.
</p>
```

```@raw html
<span id="r-identity_transform"></span>
```

## `identity_transform`

```@raw html
<h3>Description</h3>

<p>Identity transform (no constraint)
</p>


<h3>Usage</h3>

<pre><code class='language-R'>identity_transform()
</code></pre>


<h3>Value</h3>

<p>A transform specification list.
</p>
```

```@raw html
<span id="r-lower_transform"></span>
```

## `lower_transform`

```@raw html
<h3>Description</h3>

<p>Maps unconstrained <code class="reqn">y \in \mathbb{R}</code> to <code class="reqn">x = L + \exp(y)</code>.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>lower_transform(lower)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="lower">lower</code></td>
<td>
<p>Numeric lower bound.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A transform specification list.
</p>
```

```@raw html
<span id="r-upper_transform"></span>
```

## `upper_transform`

```@raw html
<h3>Description</h3>

<p>Maps unconstrained <code class="reqn">y \in \mathbb{R}</code> to <code class="reqn">x = U - \exp(y)</code>.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>upper_transform(upper)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="upper">upper</code></td>
<td>
<p>Numeric upper bound.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A transform specification list.
</p>
```

```@raw html
<span id="r-double_transform"></span>
```

## `double_transform`

```@raw html
<h3>Description</h3>

<p>Maps unconstrained <code class="reqn">y \in \mathbb{R}</code> to
<code class="reqn">x = L + (U - L) \cdot \mathrm{logistic}(y)</code>.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>double_transform(lower, upper)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="lower">lower</code></td>
<td>
<p>Numeric lower bound.</p>
</td></tr>
<tr><td><code id="upper">upper</code></td>
<td>
<p>Numeric upper bound.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A transform specification list.
</p>
```

