"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def gaussian_pdf(X, mu, sigma):
    n, d = X.shape
    I_Mat = np.eye(d)
    sigma_I = sigma * I_Mat
    det_sigma = np.linalg.det(sigma_I)
    inv_sigma = np.linalg.inv(sigma_I)
    norm_factor = 1.0 / (np.sqrt((2 * np.pi) ** d * det_sigma))
    exp_term = np.exp(-0.5 * np.sum((X - mu) @ inv_sigma * (X - mu), axis=1))
    return norm_factor * exp_term

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    X = np.array(X)
    Mus, Var, Weights = mixture
    n, k = X.shape[0], len(Weights)
    gamma = np.zeros((n, k))
    for i in range(k):
        gamma[:, i] = Weights[i] * gaussian_pdf(X, Mus[i], Var[i])
    gamma /= np.sum(gamma, axis=1, keepdims=True)

    ll = 0
    for i in range(k):
        ll += gamma[:,i]*np.log(Weights[i] * gaussian_pdf(X, Mus[i], Var[i])/gamma[:,i])

    return gamma, np.sum(ll)



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    N_Data, Dim = X.shape
    N_Cluster = post.shape[1]

    Norm_Consts = post.sum(axis=0)

    Mus = (X.transpose() @ post).transpose() / Norm_Consts.reshape(-1, 1)

    Weights = Norm_Consts / N_Data

    Var = np.zeros(N_Cluster)
    for k in range(N_Cluster):
        Temp = ((X.transpose() - Mus[k, :].reshape(-1, 1)) ** 2).sum(axis=0)

        Var[k] = \
            np.dot(Temp, post[:, k]) / (Norm_Consts[k] * Dim)

    return GaussianMixture(Mus, Var, Weights)


def log_likelihood(X, mixture, post):
    mu, var, pi = mixture
    n, k = X.shape[0], len(pi)
    ll = 0
    for i in range(k):
        ll += post[:,i]*np.log(pi[i] * gaussian_pdf(X, mu[i], var[i])/post[:,i])
    return np.sum(ll)

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
    while (old_ll is None or abs(new_ll-old_ll) > 1e-6* abs(new_ll)):
        old_ll = new_ll
        post,new_ll = estep(X, mixture)
        mixture = mstep(X, post)




    return mixture, post, new_ll

    raise NotImplementedError
