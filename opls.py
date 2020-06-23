"""
Orthogonal Projection on Latent Structure (O-PLS)

Reference:
[1] Trygg J, Wold S. Projection on Latent Structure (OPLS).
    J Chemometrics. 2002, 16, 119-128.
"""
import numpy as np
from numpy import linalg as la


def _nipals(X, y, tol=1e-10, max_iter=1000, dot=np.dot):
    """
    Non-linear Iterative Partial Least Squares
    Args
    ----
    X: np.ndarray
        Variable matrix with size n by p, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by 1.
    tol: float
        Tolerance for the convergence.
    max_iter: int
        Maximal number of iterations.

    Return
    ------
    w: np.ndarray
        Weights with size p by 1.
    u: np.ndarray
        Y-scores with size n by 1.
    c: float
        Y-weight
    t: np.ndarray
        Scores with size n by 1

    References
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.

    """
    u = y
    i = 0
    d = tol * 10
    while d > tol and i <= max_iter:
        w = dot(u, X) / dot(u, u)
        w /= la.norm(w)
        t = dot(X, w)
        c = dot(t, y) / dot(t, t)
        u_new = y * c / (c * c)
        d = la.norm(u_new - u) / la.norm(u_new)
        u = u_new
        i += 1

    return w, u, c, t


class _opls():
    """
    Orthogonal Projection on Latent Structure (O-PLS)
    """
    def __init__(self):
        """
        TODO:
        1. add arg for specifying the method for performing PLS

        """
        pass

    def fit(self, X, y, n_component=1, dot=np.dot):
        """ Fit PLS model. """
        n, p = X.shape
        # initialization
        Tortho = np.empty((n, n_component))
        Portho = np.empty((p, n_component))
        Wortho = np.empty((p, n_component))
        T = np.empty((n, n_component))
        P = np.empty((p, n_component))
        C = np.empty(n_component)

        # X-y variations
        tw = dot(y, X) / dot(y, y)
        tw /= la.norm(tw)
        # predictive scores
        tp = dot(X, tw)
        # components
        w, u, c, t = _nipals(X, y)
        p = dot(t, X) / dot(t, t)
        for nc in range(n_component):
            # orthoganol weights
            w_ortho = p - (dot(tw, p) * tw)
            w_ortho /= la.norm(w_ortho)
            # orthoganol scores
            t_ortho = dot(X, w_ortho)
            # orthoganol loadings
            p_ortho = dot(t_ortho, X) / dot(t_ortho, t_ortho)
            # update X to the residue matrix
            X -= t_ortho[:, np.newaxis] * p_ortho
            # save to matrix
            Tortho[:, nc] = t_ortho
            Portho[:, nc] = p_ortho
            Wortho[:, nc] = w_ortho
            # predictive scores
            tp -= t_ortho * dot(p_ortho, tw)
            T[:, nc] = tp
            C[nc] = dot(y, tp) / dot(tp, tp)
            
            # next component
            w, u, c, t = _nipals(X, y)
            p = dot(t, X) / dot(t, t)
            P[:, nc] = p
            

        self._Tortho = Tortho
        self._Portho = Portho
        self._Wortho = Wortho
        # covariate weights
        self._w = tw

        # coefficients and predictive scores
        self._T = T
        self._P = P
        self.coef = tw * C[:, np.newaxis]
        # calculate q2
        self.npc = n_component
        yp = self.predict(X)
        self.q2 = self._q2(y, yp)

    def predict(self, X, n_component=None, return_score=False):
        """ Predict the new coming data matrx. """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        coef = self.coef[n_component - 1]

        y = np.dot(X, coef)
        if return_score:
            return y, np.dot(X, self._w)

        return y

    def correct(self, X, n_component=None,
                return_scores=False,
                dot=np.dot):
        """
        Correction of X

        Args
        ----
        X: np.ndarray
            Data matrix with size n by c, where n is number of
            samples, and c is number of variables
        n_component: int | None
            Number of components. If is None, the number of components
            used in fitting the model is used. Default is None.
        return_scores: bool
            Return orthogonal scores. Default is False.

        Return
        X: np.ndarray
            Corrected data, with same matrix size with input X.
        T: np.ndarray
            Orthogonal score, n by n_component.

        """
        # TODO: Check X type and dimension consistencies between X and
        # scores in model.
        if n_component is None:
            n_component = self.npc

        if X.ndim == 1:
            T = np.empty(n_component)
            for nc in range(n_component):
                t = dot(X, self._Wortho[:, nc])
                X -= t * self._Portho[:, nc]
                T[nc] = t
        else:
            n, c = X.shape
            T = np.empty((n, n_component))
            # scores
            for nc in range(n_component):
                t = dot(X, self._Wortho[:, nc])
                X -= t[:, np.newaxis] * self._Portho[:, nc]
                T[:, nc] = t

        if return_scores:
            return X, T

        return X

    def q2(self, y, ypred):
        """
        prediction error Q2

        Args
        ----
        y: np.ndarray
            Original dependent variable / class labels
        ypred: np.ndarray
            Predicted dependent variable / class labels

        Return
        q2

        Reference:
        [1] Westerhuis JA, et al. Assessment of PLSDA cross validation.
            Metabolomics. 2008, 4, 81–89.
        """
        return self._q2(y, ypred)

    def covariance(self, X, t):
        """ Covariance. """
        return np.dot(t, X) / np.dot(t, t)

    def correlation(self, X, t):
        """ Correlation. """
        return np.dot(t, X) / (la.norm(t) * la.norm(X, axis=0))

    def predictive_score(self, n_component=None):
        """ Return predictive score. """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._T[:, n_component-1]

    def ortho_score(self, n_component=None):
        """ Return orthogonal score. """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._Tortho[:, n_component-1]

    def _q2(self, y, ypred):
        """ Q2. """
        return 1 - ((y - ypred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


class _pls():
    """ Partial least squares. """
    def __init__(self):
        pass

    def fit(self, X, y, n_component=2, dot=np.dot):
        """
        Fit PLS model
        Args
        ----
        X: np.ndarray
            Variable matrix with size n by p, where n number
            of samples/instances, p number of variables
        y: np.ndarray
            Dependent variable with size n by 1

        Return
        PLS object

        """
        n, r = X.shape
        # preallocation
        T = np.empty((n, n_component))
        P = np.empty((r, n_component))
        W = np.empty((r, n_component))
        C = np.empty(n_component)
        # iterate through components
        for nc in range(n_component):
            w, u, c, t = _nipals(X, y)
            # loadings
            p = dot(t, X) / dot(t, t)
            # update data matrix for next component
            X -= t[:, np.newaxis] * p
            y -= t * c
            # save to matrix
            T[:, nc] = t
            P[:, nc] = p
            W[:, nc] = w
            C[nc] = c

        # save results to matrix
        self._T = T
        self._P = P
        self._W = W
        self._C = C

        # coefficients
        _coefs = np.empty((n_component, r))
        for nc in range(n_component):
            _coefs[nc] = dot(
                dot(W[:, :nc], la.inv(dot(P[:, :nc].T, W[:, :nc]))),
                C[:nc])
        self.coef = _coefs

    def predict(self, X, n_component=None):
        """ Do prediction. """
        nc = self.coef.shape[1] - 1
        if n_component is not None and n_component < nc:
            nc = n_component - 1
        _coef = self.coef[nc]
        return np.dot(X, _coef)
