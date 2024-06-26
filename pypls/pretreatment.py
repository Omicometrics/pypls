"""
Preprocess data matrix.
"""
import numpy as np
from typing import Optional

from .core import scale_x_class, scale_x_reg


class Scaler:
    """
    A scaler to scale data matrix

    Parameters
    ----------
    scaler: str
        Method for scaling, "uv" for unit variance, "pareto" for
        pareto scaling, "mean" for mean scaling and "minmax" for
        minmax scaling. Default is "pareto".
    fit_type: str
        Whether scale data matrix based on the distribution of y
        labels.
            Unbalanced: The classes are unbalanced.
            Balanced: The classes are balanced.

    """

    def __init__(self, scaler="pareto", fit_type: str = "unbalanced"):
        if scaler == "mean":
            self.scaler_tag: int = 1
        elif scaler == "pareto":
            self.scaler_tag: int = 2
        elif scaler == "uv":
            self.scaler_tag: int = 3
        elif scaler == "minmax":
            self.scaler_tag: int = 4
        else:
            raise ValueError("Expected Scaler type 'uv', 'pareto', 'mean' or "
                             f"'minmax', got {scaler}.")

        self.fit_balanced: bool = fit_type == "balanced"

        self._center: Optional[np.ndarray] = None
        self._normalizer: Optional[np.ndarray] = None
        self._var_index: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray)\
            -> np.ndarray:
        """
        Fits a scaler model.

        Parameters
        ----------
        x: np.ndarray
            The data matrix x.
        y: np.ndarray
            The data matrix y.

        Returns
        -------

        """
        xs: np.ndarray = np.ascontiguousarray(x.copy())
        if not self.fit_balanced:
            xs, ix, center, normalizer = scale_x_class(xs, y, self.scaler_tag)
        else:
            xs, ix, center, normalizer = scale_x_reg(xs, self.scaler_tag)
        self._center = center
        self._normalizer = normalizer
        self._var_index = ix
        return xs

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
        if self.scaler_tag == 1:
            return np.ascontiguousarray(x)
        return np.ascontiguousarray(x / self._normalizer)

    @property
    def variable_index(self) -> np.ndarray:
        """ The indexes of variables used for model construction. """
        return self._var_index
