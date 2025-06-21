import warnings

import numpy as np
from IPython.display import clear_output
from scipy.stats import multivariate_normal as mnorm
from scipy.stats import norm

from .utils import Higham, check_positive_definite, heaviside, make_symmetric


class WarningDGOpt(UserWarning):
    pass


def get_bivargauss_cdf(input1: float, input2: float, corr_coef: float) -> float:
    """
    Computes cdf of a bivariate Gaussian distribution with mean zero, variance 1 and input correlation.

    Inputs:
        input1: mean of the Gaussian latent (μi).
        input2: mean of the second Gaussian latent (μj).
        corr_coef: correlation coefficient between the two Gaussian latents (ρij).

    Returns:
        float: bivariate cdf Φ2([μi, μj], Λij)
    """
    cov = np.eye(2)
    cov[1, 0], cov[0, 1] = corr_coef, corr_coef
    cdf = mnorm.cdf([input1, input2], mean=[0.0, 0.0], cov=cov)
    return cdf


def objective_function(
    data_means: np.ndarray,
    gauss_means: np.ndarray,
    data_covar: float,
    gauss_covar: float,
) -> float:
    """
    Computes the pairwise covariance eqn for root finding algorithm.

    Inputs:
        data_means: means of the binary data (m_i, m_j).
        gauss_means: means of the bivariate Gaussian (μ_i, μ_j).
        data_covar: covariance of the binary data (Σ_ij).
        gauss_covar: covariance of the bivariate Gaussian (Λ_ij).

    Returns:
        float: objective function for root finding algorithm
            Φ2([μ_i, μ_i], Λ_ij) - r_i*r_j - Σ_ij
    """
    bivar_gauss_cdf = np.mean(
        get_bivargauss_cdf(
            input1=gauss_means[0], input2=gauss_means[1], corr_coef=gauss_covar
        )
    )
    return bivar_gauss_cdf - np.prod(data_means) - data_covar


def find_root_bisection(*eqn_input, eqn=objective_function, maxiters=1000, tol=1e-10):
    """
    Finds root of input equation using the bisection algorithm.

    Inputs:
        eqn_input: inputs to the equation for which root is to be found.
        eqn: function for which root is to be found. Default is objective_function.
        maxiters: maximum number of iterations for the bisection algorithm. Default is 1000.
        tol: tolerance value for root finding. Default is 1e-10.

    Returns:
        float: root of the input equation.
    """
    λ0 = -0.99999
    λ1 = 0.99999

    f0 = eqn(*eqn_input, λ0)
    f1 = eqn(*eqn_input, λ1)

    # print('f0, f1', f0, f1)

    if np.abs(f0) < tol:
        warnings.warn(
            "Warning: f0 is already close to 0. Returning initial value.", WarningDGOpt
        )
        return λ0

    if np.abs(f1) < tol:
        warnings.warn(
            "Warning: f1 is already close to 0. Returning initial value.", WarningDGOpt
        )
        return λ1

    if f0 * f1 > tol:
        warnings.warn(
            "Warning: Both initial covariance values lie on same side of zero crossing. "
            "Setting value to 0.",
            WarningDGOpt,
        )
        λ = 0.0
        return λ

    f = np.inf
    it = 0
    while np.abs(f) > tol and it < maxiters:
        λ = (λ0 + λ1) / 2
        f = eqn(*eqn_input, λ)

        # print('λ, f(λ)', λ, f)

        if f > 0:
            λ1 = λ
        elif f < 0:
            λ0 = λ
        it += 1
    clear_output(wait=True)
    return λ


class DGModel:
    """
    Finds the parameters of the multivariate Gaussian that best fit the given binary data.
    Inputs:
        data: binary data to fit the DG model to of shape (timebins, repeats, features).
    """

    def __init__(self, data: np.ndarray) -> None:
        self.timebins, self.repeats, self.features = data.shape
        self.tril_inds = np.tril_indices(self.features, -1)
        self.data = data

    @property
    def gauss_mean(self) -> np.ndarray:
        """
        Computes mean of the multivariate Gaussian corresponding to the input binary data.
        """
        data = self.data

        mean = data.mean(1)

        # Need this to ensure inverse cdf calculation (norm.ppf()) does not break
        mean[mean == 0.0] += 1e-4
        mean[mean == 1.0] -= 1e-4

        gauss_mean = norm.ppf(mean)
        return gauss_mean

    @property
    def data_tvar_covariance(self) -> np.ndarray:
        """
        Computes covariance between observed binary vectors, averaged across timebins and repeats.
        Calculated for time-varying features (e.g. firing rates).
        """
        data = self.data

        data_norm = (data - data.mean(0)).reshape(self.timebins, -1)
        tot_covar = data_norm.T.dot(data_norm).reshape(
            self.repeats, self.features, self.repeats, self.features
        )
        inds = range(self.repeats)
        tot_covar = tot_covar[inds, :, inds, :].mean(0) / self.timebins
        return tot_covar

    @property
    def data_tfix_covariance(self) -> np.ndarray:
        """
        Computes covariance between observed binary vectors, averaged across repeats.
        Calculated for fixed features across timebins.
        """
        data = self.data
        data_norm = (data - data.mean(1)).reshape(-1, self.features)
        tot_covar = data_norm.T.dot(data_norm) / (self.timebins * self.repeats)

        return tot_covar

    def _compute_gauss_correlation(self):
        """
        Computes the correlation matrix of the multivariate Gaussian that best fits the input binary data.
        Inputs:

        Returns:
            :return: computed correlation matrix of multivariate Gaussian distribution.
        """
        data_mean = self.data.mean(1).mean(0)
        gauss_mean = self.gauss_mean
        if self.timebins > 1:
            data_covar = self.data_tvar_covariance
        else:
            data_covar = self.data_tfix_covariance

        gauss_corr = np.eye(self.features)

        # Find pairwise correlation between each unique pair of neurons
        for i, j in zip(*self.tril_inds):
            # print("Neuron pair:", i, j)
            if np.abs(data_covar[i][j]) <= 1e-10:
                warnings.warn(
                    "Data covariance is zero. Setting corresponding Gaussian dist. covariance to 0."
                )
                gauss_corr[i][j], gauss_corr[j][i] = 0.0, 0.0

            else:
                x = find_root_bisection(
                    [data_mean[i], data_mean[j]],
                    [gauss_mean[..., i], gauss_mean[..., j]],
                    data_covar[i][j],
                )
                gauss_corr[i][j], gauss_corr[j][i] = x, x

        return gauss_corr

    def get_gauss_covariance(self) -> np.ndarray:
        """
        Computes covariance matrix of the multivariate Gaussian that best fits the input binary data.
        Inputs:

        Returns:
            :return: computed covariance matrix of multivariate Gaussian distribution.
        """
        gauss_corr = self._compute_gauss_correlation()
        if not check_positive_definite(gauss_corr):
            higham = Higham()
            gauss_corr = higham.higham_correction(gauss_corr)
        gauss_cov = make_symmetric(gauss_corr)
        setattr(self, "gauss_cov", gauss_cov)
        return gauss_cov


def sample(
    gauss_mean: np.ndarray, gauss_cov: np.ndarray, n_samples: int = 1
) -> np.ndarray:
    """
    Sample from the fitted DG model.

    Inputs:
        gauss_mean: mean of the multivariate Gaussian.
        gauss_cov: covariance of the multivariate Gaussian.
        n_samples: number of samples to draw.
    Returns:
        np.ndarray: samples drawn from the fitted DG model.
    """
    samples = mnorm(mean=gauss_mean, cov=gauss_cov).rvs(size=n_samples)

    return heaviside(samples)


def pdf(gauss_mean: np.ndarray, gauss_cov: np.ndarray, x: np.ndarray) -> float | np.ndarray:
    """
    Computes the probability density function of the fitted DG model.

    Inputs:
        gauss_mean: mean of the multivariate Gaussian.
        gauss_cov: covariance of the multivariate Gaussian.
        x: input binary data to compute pdf for.
    Returns:
        np.ndarray: pdf values for the input data.
    """
    lower_limit = np.zeros_like(gauss_mean) * -np.inf
    upper_limit = np.ones_like(gauss_mean) * np.inf

    lower_limit = np.where(x == 0, upper_limit, x)
    upper_limit = np.where(x == 1, lower_limit, x)

    return mnorm.cdf(
        x=upper_limit,
        mean=gauss_mean,
        cov=gauss_cov,
        lower_limit=lower_limit,
        allow_singular=True,
    )
