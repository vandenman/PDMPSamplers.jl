# Regression tests


## Diamonds

- Stan model: `brms_diamonds_gaussian.stan`
- dataset: `diamonds.json`

Problem: Although Stan transforms the parameter space to $\mathbb{R}^d$, this doesn't mean the likelihood is nonzero everywhere. When the likelihood is zero for a particular region, the gradient is not defined. For this particular problem, some parameter values yield a gradient error when evaluated through BridgeStan. <TODO: example parameter configuration that leads to the error>.

This regression test is tested in <julia test source file>.