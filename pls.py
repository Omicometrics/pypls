import numpy as np
import numpy.linalg as la

from typing import Optional

from base import nipals


class PLS:
    """ Partial least squares. """
    def __init__(self):
        self._T: Optional[np.ndarry] = None
        self._P: Optional[np.ndarry] = None
        self._W: Optional[np.ndarry] = None
        self._C: Optional[np.ndarry] = None
        self.coef: Optional[np.ndarry] = None

    def fit(self, x, y, n_comp=None) -> None:
        """
        Fit PLS model

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n by p, where n number
            of samples/instances, p number of variables
        y: np.ndarray
            Dependent variable with size n by 1
        n_comp: int
            Number of components. Default is None, which indicates that
            smaller number between n and p will be used.

        Returns
        -------
        PLS object

        """
        n, r = x.shape
        # pre-allocation
        T = np.empty((n, n_comp))
        P = np.empty((r, n_comp))
        W = np.empty((r, n_comp))
        C = np.empty(n_comp)
        # iterate through components
        for nc in range(n_comp):
            w, u, c, t = nipals(x, y)
            # loadings
            p = np.dot(t, x) / np.dot(t, t)
            # update data matrix for next component
            x -= t[:, np.newaxis] * p
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
        # noinspection SpellCheckingInspection
        coefs = np.empty((n_comp, r))
        for nc in range(n_comp):
            coefs[nc] = np.dot(np.dot(
                W[:, :nc], la.inv(np.dot(P[:, :nc].T, W[:, :nc]))
            ), C[:nc])
        self.coef = coefs

    def predict(self, X, n_component=None):
        """ Do prediction. """
        npc = self.coef.shape[1] - 1
        if n_component is not None and n_component < npc:
            npc = n_component - 1
        coef = self.coef[npc]
        return np.dot(X, coef)

    @property
    def scores_x(self):
        """ Scores.

        Returns
        -------
        np.ndarray
            Scores

        """
        return self._T

    @property
    def loadings_x(self):
        """

        Returns
        -------
        np.ndarray
            loadings

        """
        return self._P

    @property
    def weights_y(self):
        """

        Returns
        -------
        np.ndarray
            y scores

        """
        return self._C
