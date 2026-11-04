data {
  int<lower=0> N;
  int<lower=1> D;
  matrix[N, D] X;
  array[N] int<lower=0, upper=1> y;
  real<lower=0> sd_prior;
}
parameters {
  real alpha;
  vector[D] beta;
}
model {
  alpha ~ normal(0, sd_prior);
  beta ~ normal(0, sd_prior);
  y ~ bernoulli_logit_glm(X, alpha, beta);
}