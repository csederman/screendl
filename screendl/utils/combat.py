"""
ComBat implementation (Johnson et al., 2007) for removing batch effects.
Works on log(TPM+1) gene expression data
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


def combat(
    data: pd.DataFrame,
    batch: pd.Series,
    covar: Optional[pd.DataFrame] = None,
    parametric: bool = True,
    mean_only: bool = False,
) -> pd.DataFrame:
    """
    Empirical-Bayes batch adjustment (ComBat).

    Parameters
    ----------
    data : pd.DataFrame
        Gene expression matrix, shape (n_genes, n_samples).
        Columns = samples, index = gene identifiers.
        Expected to be log(TPM+1) or any continuous log-scale values.
    batch : pd.Series
        Batch labels aligned to data.columns (e.g. cancer type strings).
    covar : pd.DataFrame, optional
        Biological covariates to protect, shape (n_samples, n_covariates).
        Index must match data.columns. Numeric; categorical vars should be
        dummy-encoded by the caller.
    parametric : bool
        If True (default), use parametric EB priors (Normal–Inverse Gamma).
        If False, use non-parametric EB (kernel density).
    mean_only : bool
        If True, correct only additive (location) batch effects and skip
        the multiplicative (scale) adjustment.

    Returns
    -------
    pd.DataFrame
        Adjusted expression matrix, same shape/index as input.
    """
    dat = data.values.astype(np.float64)  # (G, N)
    n_genes, n_samples = dat.shape
    sample_idx = data.columns
    gene_idx = data.index

    batch = batch.loc[sample_idx]  # align
    batches = batch.unique()
    n_batch = len(batches)
    batch_labels = batch.values

    if n_batch < 2:
        raise ValueError("Need ≥2 batches for ComBat adjustment.")

    # batch membership indices
    batch_idx = {b: np.where(batch_labels == b)[0] for b in batches}
    n_per_batch = np.array([len(batch_idx[b]) for b in batches])

    # build design matrix & standardize:
    # design = [batch indicators | covariates]
    # We use sum-to-zero / intercept-free batch encoding so the intercept
    # is the grand mean; covariates are appended.
    batch_design = np.zeros((n_samples, n_batch))
    for i, b in enumerate(batches):
        batch_design[batch_idx[b], i] = 1

    if covar is not None:
        covar = covar.loc[sample_idx]
        mod = covar.values.astype(np.float64)
        design = np.hstack([batch_design, mod])
    else:
        design = batch_design

    # OLS per gene: B_hat = (X'X)^-1 X' Y'  (Y is G×N, so work with Y')
    XtX_inv = np.linalg.pinv(design.T @ design)
    B_hat = (XtX_inv @ design.T @ dat.T).T  # (G, n_batch + n_cov)

    # grand mean = weighted average of batch intercepts
    grand_mean = (n_per_batch / n_samples) @ B_hat[:, :n_batch].T  # (G,)

    # covariate contribution (0 if no covariates)
    if covar is not None:
        covar_effects = dat - (batch_design @ B_hat[:, :n_batch].T).T  # remove batch part
        # but we want to keep covariate effects, so we compute them:
        covar_effects = (design[:, n_batch:] @ B_hat[:, n_batch:].T).T  # (G, N)
    else:
        covar_effects = np.zeros_like(dat)

    # standardized residuals per batch
    stand_mean = grand_mean[:, None] + covar_effects  # what we'd expect w/o batch effect

    # pooled variance per gene (across all batches, after removing design effects)
    resid = dat - (design @ B_hat.T).T
    var_pooled = np.sum(resid ** 2, axis=1) / max(n_samples - design.shape[1], 1)  # (G,)

    # standardize: Z_ij = (Y_ij - stand_mean_ij) / sqrt(var_pooled_i)
    sqrt_var = np.sqrt(var_pooled)
    sqrt_var[sqrt_var == 0] = 1e-10  # guard
    Z = (dat - stand_mean) / sqrt_var[:, None]

    # estimate raw batch parameters
    gamma_hat = np.zeros((n_batch, n_genes))   # additive
    delta_hat = np.zeros((n_batch, n_genes))   # multiplicative (variance)

    for i, b in enumerate(batches):
        idx = batch_idx[b]
        gamma_hat[i] = Z[:, idx].mean(axis=1)
        delta_hat[i] = Z[:, idx].var(axis=1, ddof=1)

    # replace NaN/0 variances (batches with n=1)
    delta_hat = np.where(np.isnan(delta_hat) | (delta_hat == 0), 1.0, delta_hat)

    # empirical Bayes shrinkage
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    for i, b in enumerate(batches):
        n_b = n_per_batch[i]

        # hyperparameters for gamma (Normal prior)
        gamma_bar = gamma_hat[i].mean()
        tau2 = gamma_hat[i].var()

        # hyperparameters for delta (Inverse Gamma prior)
        # method of moments on delta_hat[i]
        delta_mean = delta_hat[i].mean()
        delta_var = delta_hat[i].var()
        if delta_var > 0:
            # IG parameterisation: E[X] = beta/(alpha-1), Var[X] = beta^2/((a-1)^2(a-2))
            lambda_bar = (delta_mean ** 2 + 2 * delta_var) / delta_var  # alpha
            theta_bar = (delta_mean ** 3 + delta_mean * delta_var) / delta_var  # beta
        else:
            lambda_bar = 2.001
            theta_bar = delta_mean

        if parametric:
            # posterior mean for gamma_i (conjugate Normal)
            # gamma_star_ig = (tau2 * n_b * gamma_hat_ig + delta_ig * gamma_bar) / (tau2*n_b + delta_ig)
            gamma_star[i] = _post_gamma(gamma_hat[i], gamma_bar, tau2,
                                        delta_hat[i], n_b)

            if not mean_only:
                # posterior mean for delta under IG prior
                delta_star[i] = _post_delta_parametric(
                    delta_hat[i], n_b, lambda_bar, theta_bar
                )
            else:
                delta_star[i] = np.ones(n_genes)
        else:
            # non-parametric EB
            gamma_star[i] = _post_gamma(gamma_hat[i], gamma_bar, tau2,
                                        delta_hat[i], n_b)
            if not mean_only:
                delta_star[i] = _post_delta_nonparametric(
                    delta_hat[i], n_b
                )
            else:
                delta_star[i] = np.ones(n_genes)

    # adjust data
    adjusted = np.zeros_like(dat)
    for i, b in enumerate(batches):
        idx = batch_idx[b]
        denom = np.sqrt(delta_star[i])
        denom[denom == 0] = 1e-10
        adjusted[:, idx] = (
            stand_mean[:, idx]
            + sqrt_var[:, None]
            * (Z[:, idx] - gamma_star[i][:, None]) / denom[:, None]
        )

    return pd.DataFrame(adjusted, index=gene_idx, columns=sample_idx)


def _post_gamma(gamma_hat_i, gamma_bar, tau2, delta_hat_i, n_b):
    """Posterior mean of gamma under Normal prior."""
    if tau2 == 0:
        return np.full_like(gamma_hat_i, gamma_bar)
    return (tau2 * n_b * gamma_hat_i + delta_hat_i * gamma_bar) / (
        tau2 * n_b + delta_hat_i
    )


def _post_delta_parametric(delta_hat_i, n_b, lambda_bar, theta_bar):
    """Posterior mean of delta under Inverse-Gamma prior (parametric EB)."""
    # posterior IG: alpha_post = lambda_bar + n_b/2
    #               beta_post  = theta_bar + 0.5 * sum_of_sq  (approx as n_b/2 * delta_hat)
    # E[delta | data] = beta_post / (alpha_post - 1)
    alpha_post = lambda_bar + n_b / 2.0
    beta_post = theta_bar + n_b * delta_hat_i / 2.0
    out = beta_post / (alpha_post - 1.0)
    out[out <= 0] = delta_hat_i[out <= 0]  # safety
    return out


def _post_delta_nonparametric(delta_hat_i, n_b, n_grid=500):
    """Non-parametric EB posterior for delta via kernel density."""
    G = len(delta_hat_i)
    # use log(delta) for KDE
    log_d = np.log(delta_hat_i.clip(1e-10))
    bw = 0.9 * min(log_d.std(), np.subtract(*np.percentile(log_d, [75, 25])) / 1.34) * G ** (-0.2)
    if bw <= 0:
        bw = 0.5

    grid = np.linspace(log_d.min() - 3 * bw, log_d.max() + 3 * bw, n_grid)
    # KDE on grid
    density = np.zeros(n_grid)
    for val in log_d:
        density += norm.pdf(grid, loc=val, scale=bw)
    density /= G

    # for each gene, compute weighted posterior mean
    delta_star = np.zeros(G)
    for g in range(G):
        # likelihood: chi-sq  => log-lik of delta given data
        log_lik = (n_b / 2.0 - 1) * grid - n_b * delta_hat_i[g] / (2.0 * np.exp(grid))
        log_weights = log_lik + np.log(density + 1e-300)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum() + 1e-300
        delta_star[g] = np.sum(weights * np.exp(grid))

    delta_star[delta_star <= 0] = delta_hat_i[delta_star <= 0]
    return delta_star


def remove_cancer_type_effects(
    expr: pd.DataFrame,
    cancer_type: pd.Series,
    covariates: Optional[pd.DataFrame] = None,
    parametric: bool = True,
    mean_only: bool = False,
) -> pd.DataFrame:
    """
    Remove cancer-type-specific gene expression patterns from log(TPM+1) data.

    Parameters
    ----------
    expr : pd.DataFrame
        log(TPM+1) expression, shape (n_genes, n_samples).
    cancer_type : pd.Series
        Cancer type label per sample, indexed by sample ID.
    covariates : pd.DataFrame, optional
        Biological covariates to protect (e.g. drug response, mutation status).
        Must be numeric. Categorical variables should be dummy-encoded first.
    parametric : bool
        Parametric (default) or non-parametric EB.
    mean_only : bool
        If True, only correct location shifts (useful when you believe
        cancer-type differences are mostly mean shifts in expression).

    Returns
    -------
    pd.DataFrame
        Adjusted expression matrix.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> n_genes, n_samples = 500, 120
    >>> genes = [f"gene_{i}" for i in range(n_genes)]
    >>> samples = [f"sample_{i}" for i in range(n_samples)]
    >>>
    >>> # simulate 3 cancer types with strong batch effects
    >>> ct_labels = np.array(["BRCA"]*40 + ["LUAD"]*40 + ["COAD"]*40)
    >>> cancer_type = pd.Series(ct_labels, index=samples)
    >>>
    >>> # base expression + cancer-type shift + noise
    >>> base = np.random.randn(n_genes, n_samples) * 0.5 + 6.0
    >>> batch_shift = np.zeros((n_genes, n_samples))
    >>> for i, ct in enumerate(["BRCA", "LUAD", "COAD"]):
    ...     idx = np.where(ct_labels == ct)[0]
    ...     batch_shift[:, idx] = np.random.randn(n_genes, 1) * 1.5
    >>> expr_raw = pd.DataFrame(base + batch_shift, index=genes, columns=samples)
    >>>
    >>> expr_adj = remove_cancer_type_effects(expr_raw, cancer_type)
    >>> print(f"Before: batch var = {_batch_variance(expr_raw, cancer_type):.3f}")
    >>> print(f"After:  batch var = {_batch_variance(expr_adj, cancer_type):.3f}")
    """
    return combat(
        data=expr,
        batch=cancer_type,
        covar=covariates,
        parametric=parametric,
        mean_only=mean_only,
    )


def _batch_variance(expr, batch):
    """Mean across-gene variance of batch means (diagnostic)."""
    means = []
    for b in batch.unique():
        idx = batch[batch == b].index
        means.append(expr[idx].mean(axis=1).values)
    return np.var(np.column_stack(means), axis=1).mean()

