"""
Preprocess data matrix.
"""
import numpy as np
from typing import Optional


class Scaler:
    """
    A scaler to scale data matrix

    Parameters
    ----------
    scaler: str
        Method for scaling, "uv" for unit variance, "pareto" for
        pareto scaling, "mean" for mean scaling and "minmax" for
        minmax scaling. Default is "pareto".
    """

    def __init__(self, scaler="pareto"):
        if scaler == "uv":
            self.scaler = self._autoscaling
        elif scaler == "pareto":
            self.scaler = self._paretoscaling
        elif scaler == "mean":
            self.scaler = self._meancentering
        elif scaler == "minmax":
            self.scaler = self._minmaxscaling

        self._center: Optional[np.ndarray] = None
        self._normalizer: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> np.ndarray:
        """
        Fit scaler model.
        :param x: variable matrix for scaling and parameter setup
        :return: Scaler object and scaled X
        """
        center, normalizer, xscale = self.scaler(x)
        self._center = center
        self._normalizer = normalizer
        return xscale

    def scale(self, x) -> np.ndarray:
        """
        Scale the x based on the parameters obtained in fit.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.

        Returns
        -------
            np.ndarray
            scaled x
        """
        x = x - self._center
        return x if self._normalizer is None else x / self._normalizer

    @staticmethod
    def _autoscaling(x) -> tuple:
        """
        Mean center and unit variance scaling.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples and p variables.

        Returns
        -------
            np.ndarray
            scaled x
        """
        center = x.mean(axis=0)
        normalizer = x.std(axis=0)
        return center, normalizer, (x - center) / normalizer

    @staticmethod
    def _paretoscaling(x) -> tuple:
        """
        Pareto scaling

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples and p variables.

        Returns
        -------
            np.ndarray
            scaled x
        """
        center = x.mean(axis=0)
        normalizer = np.sqrt(x.std(axis=0))
        return center, normalizer, (x - center) / normalizer

    def _meancentering(x) -> tuple:
        """
        Mean center

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples and p variables.

        Returns
        -------
            np.ndarray
            scaled x
        """
        center = x.mean(axis=0)
        return center, None, x - center

    def _minmaxscaling(x) -> tuple:
        """
        Min-max scaling to scale each variable into range 0 and 1.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples and p variables

        Returns
        -------
            np.ndarray
            scaled x
        """
        center = x.min(axis=0)
        normalizer = x.max(axis=0) - x.min(axis=0)
        return center, normalizer, (x - center) / normalizer
