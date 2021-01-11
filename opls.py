"""
Orthogonal Projection on Latent Structure (O-PLS)
"""
import numpy as np
from numpy import linalg as la
from typing import Optional
from base import nipals


class OPLS:
    """
    Orthogonal Projection on Latent Structure (O-PLS).
    Methods
    ----------
    predictive_scores: np.ndarray
        First predictive score.
    predictive_loadings: np.ndarray
        Predictive loadings.
    weights_y: np.ndarray
        y weights.
    orthogonal_loadings: np.ndarray
        Orthogonal loadings.
    orthogonal_scores: np.ndarray
        Orthogonal scores.
    """
    def __init__(self):
        """
        TODO:
            1. add arg for specifying the method for performing PLS

        """
        # orthogonal score matrix
        self._Tortho: Optional[np.ndarray] = None
        # orthogonal loadings
        self._Portho: Optional[np.ndarray] = None
        # loadings
        self._Wortho: Optional[np.ndarray] = None
        # covariate weights
        self._w: Optional[np.ndarray] = None

        # predictive scores
        self._T: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        # coefficients
        self.coef: Optional[np.ndarray] = None
        # total number of components
        self.npc: Optional[int] = None

    def fit(self, x, y, n_comp=None):
        """
        Fit PLS model.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.
        y: np.ndarray
            Dependent matrix with size n samples by 1, or a vector
        n_comp: int
            Number of components, default is None, which indicates that
            largest dimension which is smaller value between n and p
            will be used.

        Returns
        -------
        OPLS object

        Reference
        ---------
        [1] Trygg J, Wold S. Projection on Latent Structure (OPLS).
            J Chemometrics. 2002, 16, 119-128.
        [2] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable
            regression (LVR) method with a integral OSC filter.
            J Chemometrics. 2003, 17, 53-64.

        """
        n, p = x.shape
        npc = min(n, p)
        if n_comp is not None and n_comp < npc:
            npc = n_comp

        # initialization
        Tortho = np.empty((n, npc))
        Portho = np.empty((p, npc))
        Wortho = np.empty((p, npc))
        T, P, C = np.empty((n, npc)), np.empty((p, npc)), np.empty(npc)

        # X-y variations
        tw = np.dot(y, x) / np.dot(y, y)
        tw /= la.norm(tw)
        # predictive scores
        tp = np.dot(x, tw)
        # components
        w, u, _, t = nipals(x, y)
        p = np.dot(t, x) / np.dot(t, t)
        for nc in range(npc):
            # orthoganol weights
            w_ortho = p - (np.dot(tw, p) * tw)
            w_ortho /= la.norm(w_ortho)
            # orthogonal scores
            t_ortho = np.dot(x, w_ortho)
            # orthogonal loadings
            p_ortho = np.dot(t_ortho, x) / np.dot(t_ortho, t_ortho)
            # update X to the residue matrix
            x -= t_ortho[:, np.newaxis] * p_ortho
            # save to matrix
            Tortho[:, nc] = t_ortho
            Portho[:, nc] = p_ortho
            Wortho[:, nc] = w_ortho
            # predictive scores
            tp -= t_ortho * np.dot(p_ortho, tw)
            T[:, nc] = tp
            C[nc] = np.dot(y, tp) / np.dot(tp, tp)

            # next component
            w, u, _, t = nipals(x, y)
            p = np.dot(t, x) / np.dot(t, t)
            P[:, nc] = p

        self._Tortho = Tortho
        self._Portho = Portho
        self._Wortho = Wortho
        # covariate weights
        self._w = tw

        # coefficients and predictive scores
        self._T = T
        self._P = P
        self._C = C
        self.coef = tw * C[:, np.newaxis]

        self.npc = npc

    def predict(self, X, n_component=None, return_scores=False):
        """
        Predict the new coming data matrix.
        Parameters
        ----------
        X: np.ndarray
            Variable matrix with size n samples by p variables.
        n_component: int | None
            Number of components.
        return_scores: bool
            Whether the scores should be returned.

        Returns
        -------
        y: np.ndarray
            Predicted scores for classification.
        score: np.ndarray
            Predictive scores.
        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        coef = self.coef[n_component - 1]

        y = np.dot(X, coef)
        if return_scores:
            return y, np.dot(X, self._w)

        return y

    def correct(self, x, n_component=None, return_scores=False, dot=np.dot):
        """
        Correction of X

        Parameters
        ----------
        x: np.ndarray
            Data matrix with size n by c, where n is number of
            samples, and c is number of variables
        n_component: int | None
            Number of components. If is None, the number of components
            used in fitting the model is used. Default is None.
        return_scores: bool
            Return orthogonal scores. Default is False.

        Returns
        -------
        xc: np.ndarray
            Corrected data, with same matrix size with input X.
        t: np.ndarray
            Orthogonal score, n by n_component.

        """
        # TODO: Check X type and dimension consistencies between X and
        #       scores in model.
        xc = x.copy()
        if n_component is None:
            n_component = self.npc

        if xc.ndim == 1:
            t = np.empty(n_component)
            for nc in range(n_component):
                t_ = dot(xc, self._Wortho[:, nc])
                xc -= t_ * self._Portho[:, nc]
                t[nc] = t_
        else:
            n, c = xc.shape
            t = np.empty((n, n_component))
            # scores
            for nc in range(n_component):
                t_ = dot(xc, self._Wortho[:, nc])
                xc -= t_[:, np.newaxis] * self._Portho[:, nc]
                t[:, nc] = t_

        if return_scores:
            return xc, t

        return xc

    def predictive_score(self, n_component=None):
        """
        Parameters
        ----------
        n_component: int
            The component number.

        Returns
        -------
        np.ndarray
            The first predictive score.

        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._T[:, n_component-1]

    def ortho_score(self, n_component=None):
        """

        Parameters
        ----------
        n_component: int
            The component number.

        Returns
        -------
        np.ndarray
            The first orthogonal score.

        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._Tortho[:, n_component-1]

    @property
    def predictive_scores(self):
        """ Orthogonal loadings. """
        return self._T

    @property
    def predictive_loadings(self):
        """ Predictive loadings. """
        return self._P

    @property
    def weights_y(self):
        """ y scores. """
        return self._C

    @property
    def orthogonal_loadings(self):
        """ Orthogonal loadings. """
        return self._Portho

    @property
    def orthogonal_scores(self):
        """ Orthogonal scores. """
        return self._Tortho
