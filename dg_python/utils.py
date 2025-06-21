import numpy as np
from scipy.stats import multivariate_normal as mnorm
from numpy.linalg import LinAlgError
import warnings


def heaviside(input: np.ndarray, center: float = 0) -> np.ndarray:
    """
    Implements sgn(_input - center).
    
    Args:
        input: Input array.
        center: Center value for the Heaviside function. 
            Default is 0.

    Returns:
        np.ndarray: Array of the same shape as input, with 1 
            where input > center, and 0 otherwise.
    """
    spikes = np.zeros_like(input)
    spikes[input > center] = 1.
    return spikes


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """
    Converts input covariance matrix into correlation matrix.
    
    Args:
        cov: Covariance matrix.

    Returns:
        np.ndarray: Correlation matrix of size.
    """
    std = np.sqrt(np.diag(cov))
    std_mat = np.outer(std, std)
    return cov / (std_mat + 1e-8)


def make_symmetric(M: np.ndarray) -> np.ndarray:
    """
    Makes input matrix symmetric, if it is non-symmetric.
    
    Args:
        M: Input matrix.
    Returns:
        np.ndarray: Symmetric matrix.
    """
    M_copy = M
    if np.any(M != M.T):
        tril_inds = np.tril_indices(len(M), -1)
        M_copy[tril_inds] = M[tril_inds[1], tril_inds[0]].flatten()
    return M_copy


def check_positive_definite(cov: np.ndarray) -> bool:
        """
        Checks if input covariance matrix is positive definite.
        
        Args:
            cov: Covariance matrix to check.
        Returns:
            bool: True if the covariance matrix is positive definite, 
                False otherwise.
        """
        try:
            np.linalg.cholesky(cov)
            return True
        except LinAlgError:
            return False


class Higham:
    """ Converts an input symmetric matrix M into a positive semi-definite matrix A using the Higham iterative
        projection algorithm to minimize the Frobenius norm between A and M.
        Reference: NJ Higham, Computing the nearest correlation matrix - a problem from finance, IMA Journal of
        Numerical Analysis, 2002

        Inputs:
        maxiters: max. number of iterations for iterative projection algorithm. Default is 100,000.
        tol: tolerance value for Frobenius norm. Default is 1e-10.
    """

    def __init__(self, maxiters: float=1e5, tol: float=1e-10):
        self.maxiters = maxiters
        self.tol = tol

    def projection_S(self, M: np.ndarray) -> np.ndarray:
        eigval, eigvec = np.linalg.eig(M)
        eigval[eigval < 0.] = 0.
        return eigvec.dot(np.diag(eigval).dot(eigvec.T))

    def projection_U(self, M: np.ndarray) -> np.ndarray:
        U = np.diag(np.diag(M - np.eye(len(M))))
        return M - U

    def higham_correction(self, M: np.ndarray) -> np.ndarray:

        it = 0
        DS = 0.
        Yo = M
        Xo = M
        delta = np.inf
        # triu_inds = np.triu_indices(len(cov), 1)

        while (it < self.maxiters) and (delta > self.tol):
            R = Yo - DS
            Xn = self.projection_S(R)
            DS = Xn - R
            Yn = self.projection_U(Xn)

            del_x = max(np.abs(Xn - Xo).sum(1)) / max(np.abs(Xn).sum(1))
            del_y = max(np.abs(Yn - Yo).sum(1)) / max(np.abs(Yn).sum(1))
            del_xy = max(np.abs(Yn - Xn).sum(1)) / max(np.abs(Yn).sum(1))
            delta = max(del_x, del_y, del_xy)
            Xo = Xn
            Yo = Yn

            it += 1
        if it == self.maxiters:
            warnings.warn("Iteration limit reached without convergence.")
            print('Frobenius norm:', del_x, del_y, del_xy)

        eigvals, eigvec = np.linalg.eig(Yn)
        if min(eigvals) < 0:
            warnings.warn("Higham corrected matrix was not positive definite. Converting into pd matrix.",
                          )
            eigvals[eigvals < 0.] = 1e-6
            Yn = eigvec.dot(np.diag(eigvals).dot(eigvec.T))
            Yn = cov_to_corr(Yn)
            Yn = 0.5 * (Yn + Yn.T)

        return Yn.real

