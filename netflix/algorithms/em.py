"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from naive_em import gaussian_pdf


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    # Calculate the log of the probabilities
    log_post = np.zeros((n, K))

    for k in range(K):
        # Mask the missing entries in X
        mask = X > 0

        # Calculate log of Gaussian density only on observed data
        diff = X - mixture.mu[k]
        log_prob = -0.5 * np.sum(((diff ** 2) / mixture.var[k]) * mask, axis=1)

        # Adding log of the mixture weights and constants
        log_prob -= 0.5 * np.sum(mask, axis=1) * np.log(2 * np.pi * mixture.var[k])
        log_prob += np.log(mixture.p[k] + 1e-16)

        log_post[:, k] = log_prob

    # Log-sum-exp for normalization
    logsum = logsumexp(log_post, axis=1, keepdims=True)
    log_likelihood = np.sum(logsum)
    log_post -= logsum  # Subtract logsum for normalization

    # Convert log_post to post (posterior probabilities)
    post = np.exp(log_post)

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # New mixing probabilities
    n_k = np.sum(post, axis=0)  # (K,)
    new_p = n_k / n

    # New means calculation
    new_mu = np.zeros((K, d))
    for k in range(K):
        new_mu[k] = np.sum(post[:, k, None] * (X>0) * X, axis=0) / (np.sum(post[:, k, None] * (X > 0), axis=0) + 1e-16)

    # New variances calculation
    new_var = np.zeros(K)
    for k in range(K):
        diff = X - new_mu[k]
        mask = X > 0
        new_var[k] = np.sum(post[:, k] * np.sum((diff ** 2) * mask, axis=1)) / (
                    np.sum(post[:, k] * np.sum(mask, axis=1)) + 1e-16)
        new_var[k] = max(new_var[k], min_variance)

    # Return updated GaussianMixture
    return GaussianMixture(mu=new_mu, var=new_var, p=new_p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = None
    new_ll = None
    while (old_ll is None or abs(new_ll - old_ll) > 1e-6 * abs(new_ll)):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        mixture = mstep(X, post,mixture)

    return mixture, post, new_ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """

    n, d = X.shape
    K = mixture.mu.shape[0]
    X_pred = np.copy(X)
    post = np.ones((n, K)) / K
    new_mixture, new_post, ll_final = run(X, mixture, post)

    for i in range(n):
        for j in range(d):
            if X[i, j] == 0:  # Missing entry
                # Compute the expected value for the missing entry
                X_pred[i, j] = np.dot(new_post[i], new_mixture.mu[:, j])

    return X_pred
