functions {
  matrix covariance_2(vector log_sd, real z_rho) {
    vector[2] sd = exp(log_sd);
    real rho = tanh(z_rho);
    matrix[2, 2] Sigma;
    Sigma[1, 1] = square(sd[1]);
    Sigma[2, 2] = square(sd[2]);
    Sigma[1, 2] = sd[1] * sd[2] * rho;
    Sigma[2, 1] = Sigma[1, 2];
    return Sigma;
  }

  matrix riclpm_covariance(matrix A, matrix Sigma_ri, matrix Sigma_w1,
                           matrix Sigma_e, int T, real jitter) {
    int P = rows(A);
    array[T, T] matrix[P, P] Cw;
    matrix[P * T, P * T] Sigma_z = rep_matrix(0.0, P * T, P * T);

    for (t in 1:T) for (s in 1:T) Cw[t, s] = rep_matrix(0.0, P, P);
    Cw[1, 1] = Sigma_w1;
    for (t in 2:T) {
      Cw[t, t] = A * Cw[t - 1, t - 1] * A' + Sigma_e;
      for (s in 1:(t - 1)) {
        Cw[t, s] = A * Cw[t - 1, s];
        Cw[s, t] = Cw[t, s]';
      }
    }
    for (t in 1:T) for (s in 1:T) {
      Sigma_z[((t - 1) * P + 1):(t * P),
              ((s - 1) * P + 1):(s * P)] = Sigma_ri + Cw[t, s];
    }
    for (j in 1:(P * T)) Sigma_z[j, j] += jitter;
    return Sigma_z;
  }

  real multi_normal_sufficient_lp(int N, vector z_sum,
                                  matrix z_crossprod, vector mu, matrix L) {
    int D = rows(mu);
    matrix[D, D] L_inv = mdivide_left_tri_low(
      L, diag_matrix(rep_vector(1.0, D)));
    matrix[D, D] whitened_crossprod =
      mdivide_left_tri_low(L, z_crossprod);
    vector[D] whitened_sum = mdivide_left_tri_low(L, z_sum);
    vector[D] whitened_mu = mdivide_left_tri_low(L, mu);
    real quadratic = dot_product(to_vector(L_inv),
                                 to_vector(whitened_crossprod))
      - 2.0 * dot_product(whitened_mu, whitened_sum)
      + N * dot_self(whitened_mu);
    return -0.5 * N * D * log(2.0 * pi())
      - N * sum(log(diagonal(L))) - 0.5 * quadratic;
  }
}

data {
  int<lower=1> N;
  int<lower=2, upper=2> P;
  int<lower=3> T;
  vector[P * T] z_sum;
  matrix[P * T, P * T] z_crossprod;
  real<lower=0> jitter;
  vector[2] slab_mean;
  cov_matrix[2] slab_cov;
  real<lower=0> mean_prior_sd;
  real<lower=0> lag_prior_sd;
  real log_sd_prior_mean;
  real<lower=0> log_sd_prior_sd;
  real<lower=0> rho_prior_sd;
}

parameters {
  vector[P * T] wave_mean;
  vector[P] autoregressive;
  vector[2] cross_lagged;
  vector[P] log_sd_ri;
  vector[P] log_sd_w1;
  vector[P] log_sd_e;
  real z_rho_ri;
  real z_rho_w1;
  real z_rho_e;
}

model {
  matrix[P, P] A;
  matrix[P, P] Sigma_ri = covariance_2(log_sd_ri, z_rho_ri);
  matrix[P, P] Sigma_w1 = covariance_2(log_sd_w1, z_rho_w1);
  matrix[P, P] Sigma_e = covariance_2(log_sd_e, z_rho_e);
  matrix[P * T, P * T] L_z;

  A[1, 1] = autoregressive[1];
  A[2, 2] = autoregressive[2];
  A[1, 2] = cross_lagged[1];
  A[2, 1] = cross_lagged[2];
  L_z = cholesky_decompose(
    riclpm_covariance(A, Sigma_ri, Sigma_w1, Sigma_e, T, jitter));

  wave_mean ~ normal(0, mean_prior_sd);
  autoregressive ~ normal(0, lag_prior_sd);
  cross_lagged ~ multi_normal(slab_mean, slab_cov);
  log_sd_ri ~ normal(log_sd_prior_mean, log_sd_prior_sd);
  log_sd_w1 ~ normal(log_sd_prior_mean, log_sd_prior_sd);
  log_sd_e ~ normal(log_sd_prior_mean, log_sd_prior_sd);
  z_rho_ri ~ normal(0, rho_prior_sd);
  z_rho_w1 ~ normal(0, rho_prior_sd);
  z_rho_e ~ normal(0, rho_prior_sd);

  target += multi_normal_sufficient_lp(
    N, z_sum, z_crossprod, wave_mean, L_z);
}
