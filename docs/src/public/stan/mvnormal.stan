data {
  int<lower=1> N;
  vector[N] mu;
  matrix[N, N] sigma;
}
parameters {
  vector[N] x;
}
model {
  x ~ multi_normal(mu, sigma);
}
