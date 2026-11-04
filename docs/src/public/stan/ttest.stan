data {
  int<lower=2> nx;
  int<lower=2> ny;
  vector[nx] x;
  vector[ny] y;
  real<lower=0> rscale; // e.g. 0.707 for "medium"
}
parameters {
  real mu;                 // grand mean (location)
  real delta;              // standardized effect size (delta)
  real<lower=0> sigma;     // common SD
}
transformed parameters {
  real mu1 = mu - 0.5 * delta * sigma;
  real mu2 = mu + 0.5 * delta * sigma;
}
model {
  // Jeffreys prior on sigma: p(sigma) \propto 1/sigma
  target += -log(sigma);

  // Cauchy (JZS) prior on delta: student_t(1, 0, rscale)
  target += student_t_lpdf(delta | 1, 0, rscale);

  x ~ normal(mu1, sigma);
  y ~ normal(mu2, sigma);
}