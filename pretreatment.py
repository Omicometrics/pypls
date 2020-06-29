"""
Preprocess data matrix.
"""
import numpy as np


class Scaler:
    """
    A scaler to scale data matrix
    """

    def __init__(self, scaler: str = "pareto"):
        if scaler == "uv":
            self.scaler = self._autoscaling
        elif scaler == "pareto":
            self.scaler = self._paretoscaling
        elif scaler == "mean":
            self.scaler = self._meancentering
        elif scaler == "minmax":
            self.scaler = self._minmaxscaling

        self._center: np.ndarray = None
        self._normalizer: np.ndarray = None

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

    def scale(self, x: np.ndarray) -> np.ndarray:
        """
        Scale the x based on the parameters obtained in fit
        :param x: variable matrix with size n samples by p variables
        :return: scaled x
        """
        x = x - self._center
        return x if self._normalizer is None else x / self._normalizer

    def _autoscaling(self, x: np.ndarray) -> tuple:
        """
        Mean center and unit variance scaling
        :param x: variable matrix with size n samples and p variables
        :return: scaled x
        """
        center = x.mean(axis=0)
        normalizer = x.std(axis=0)
        return center, normalizer, (x - center) / normalizer

    def _paretoscaling(self, x: np.ndarray) -> tuple:
        """
        Pareto scaling
        :param x: variable matrix with size n samples and p variables
        :return: scaled x
        """
        center = x.mean(axis=0)
        normalizer = np.sqrt(x.std(axis=0))
        return center, normalizer, (x - center) / normalizer

    def _meancentering(x: np.ndarray) -> tuple:
        """
        Mean center
        :param x: variable matrix with size n samples and p variables
        :return: scaled x
        """
        center = x.mean(axis=0)
        return center, None, x - center

    def _minmaxscaling(x: np.ndarray) -> tuple:
        """
        Min-max scaling to scale each variable into range 0 and 1
        :param x: variable matrix with size n samples and p variables
        :return: scaled x
        """
        center = x.min(axis=0)
        normalizer = x.max(axis=0) - x.min(axis=0)
        return center, normalizer, (x - center) / normalizer
