import numpy as np
import numpy.linalg as la
import typing


def nipals(x, y, tol=1e-10, max_iter=1000) -> typing.Tuple:
    """
    Non-linear Iterative Partial Least Squares

    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by p, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by 1.
    tol: float
        Tolerance for the convergence.
    max_iter: int
        Maximal number of iterations.

    Returns
    -------
    w: np.ndarray
        Weights with size p by 1.
    u: np.ndarray
        Y-scores with size n by 1.
    c: float
        Y-weight
    t: np.ndarray
        Scores with size n by 1

    References
    ----------
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.

    """
    u = y
    i = 0
    d = tol * 10
    while d > tol and i <= max_iter:
        w = np.dot(u, x) / np.dot(u, u)
        w /= la.norm(w)
        t = np.dot(x, w)
        c = np.dot(t, y) / np.dot(t, t)
        u_new = y * c / (c * c)
        d = la.norm(u_new - u) / la.norm(u_new)
        u = u_new
        i += 1

    return w, u, c, t
