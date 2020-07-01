"""
Perform cross validation.
"""
import collections
import typing
import numpy as np
import numpy.linalg as la

import pretreatment
from pls import PLS
from opls import OPLS


class CrossValidation:
    """
    Stratified cross validation

    Parameters:
    ----------
    estimator: str
        Estimator indicates algorithm for model construction.
        Values can be "pls" for PLS and "opls" for OPLS. Default
        is "opls".
    kfold: int
        k fold cross validation. if k equals to len(X), leave one out
        cross validation will be performed. Default is 10.
    scaler: str
        Scaler for scaling data matrix. Valid values are "uv" for
        zero-mean-unit-variance scaling, "pareto" for Pareto scaling,
        "minmax" for Min-Max scaling and "mean" for mean centering.
        Default is "pareto".

    Returns
    -------
    CrossValidation object

    """
    def __init__(self, estimator="opls", kfold=10, scaler="pareto") -> None:
        # number of folds
        self.kfold = kfold
        # estimator
        if estimator == "pls":
            self.estimator = PLS()
        elif estimator == "opls":
            self.estimator = OPLS()
        self.estimator_id = estimator
        # scaler
        self.scaler = pretreatment.Scaler(scaler=scaler)

        # initialize other attributes, but should be HIDDEN
        self._ypred: np.ndarray = None
        self._Tortho: np.ndarray = None
        self._Tpred: np.ndarray = None
        self._ssx: dict = None
        self._ssy: list = None
        self._pressy: np.ndarray = None
        self._n: int = None
        self._pcv: dict = None
        self._opt_component: int = None
        self._mis_classifications: list = None
        self._q2: np.ndarray = None
        self._npc0: int = None
        self._x: np.ndarray = None
        self.y: np.ndarray = None
        self.groups: dict = None

    def fit(self, x, y) -> None:
        """
        Fitting variable matrix X

        Parameters
        ----------
        x : np.ndarray
            Variable matrix with size n samples by p variables.
        y : np.ndarray | list
            Dependent matrix with size n samples by 1. The values in
            this vector must be 0 and 1, otherwise the classification
            performance will be wrongly concluded.

        Returns
        -------
        CrossValidation object

        """
        # TODO: Check dimension consistencies between X and y.
        # set the labels in y to 0 and 1, and name the groups using
        # the labels in y
        y = self._reset_y(y)
        # matrix dimension
        n, p = x.shape
        # max number of principal components
        npc0 = min(n, p)
        # preallocation
        ssx = collections.defaultdict(lambda: collections.defaultdict(list))
        ssy = []
        ypred, pressy = np.zeros((n, npc0)), np.zeros((n, npc0))
        tortho, tpred = np.zeros((n, npc0)), np.zeros((n, npc0))
        pcv = collections.defaultdict(list)
        for train_index, test_index in self._split(y):
            xtr, xte = x[train_index], x[test_index]
            ytr, yte = y[train_index], y[test_index]

            # scale matrix
            xtr_scale = self.scaler.fit(xtr)
            xte_scale = self.scaler.scale(xte)
            ytr_scale = self.scaler.fit(ytr)
            yte_scale = self.scaler.scale(yte)

            # variances
            ssy_tot = (yte_scale ** 2).sum()
            ssx_tot = (xte_scale ** 2).sum()

            # fit the model
            npc = min(xtr.shape)
            self.estimator.fit(xtr_scale.copy(), ytr_scale, n_comp=npc)
            if npc < npc0:
                npc0 = npc

            # do prediction iterating through components
            for k in range(1, npc+1):
                # if OPLS is used, the test matrix should be corrected to
                # remove orthogonal components
                if self.estimator_id == "opls":
                    xte_corr, tcorr = self.estimator.correct(
                        xte_scale, n_component=k, return_scores=True
                    )
                    # prediction
                    yp_k, tp_k = self.estimator.predict(
                        xte_corr, n_component=k, return_scores=True
                    )

                    # save the parameters for model quality assessments
                    # Orthogonal and predictive scores
                    if xte_scale.ndim == 1:
                        tortho[test_index, k-1] = tcorr[0]
                    else:
                        tortho[test_index, k-1] = tcorr[:, 0]
                    tpred[test_index, k-1] = tp_k

                    # sum of squares
                    ssx[k]["corr"].append((xte_corr ** 2).sum())
                    xte_ortho = np.dot(
                        tcorr, self.estimator.orthogonal_loadings[:, :k].T
                    )
                    ssx[k]["xyo"].append((xte_ortho ** 2).sum())
                    ssx[k]["total"].append(ssx_tot)

                    # covariances from fitting
                    tp = self.estimator.predictive_scores[:, k-1]
                    pcv[k].append(np.dot(tp, xtr_scale) / (tp ** 2).sum())

                else:
                    # prediction
                    yp_k = self.estimator.predict(xte_scale, n_component=k)

                # predicted y
                ypred[test_index, k-1] = yp_k
                pressy[test_index, k-1] = (yp_k - yte_scale) ** 2

            ssy.append(ssy_tot)

        # save metrics
        self._ypred = ypred[:, :npc0]
        self._pressy = pressy[:, :npc0]
        self._ssy = sum(ssy)
        self._n = n
        self._npc0 = npc0
        self._x = x
        self.y = y

        # opls specific metrics
        if self.estimator_id == "opls":
            self._Tortho = tortho[:, :npc0]
            self._Tpred = tpred[:, :npc0]
            self._ssx = ssx
            self._pcv = pcv

        # summarize cross validation results
        self._summary_cv()
        # refit for a final model
        self._create_optimal_model(x, y)

    def predict(self, x) -> np.ndarray:
        """Do prediction using optimal model.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.

        Returns
        -------
        np.ndarray
            Predictions for the x

        """
        # TODO: check the dimension consistencies between the training
        #       data and the input data matrix.
        npc = self._opt_component + 1
        # scale the matrix
        x = self.scaler.scale(x)
        if self.estimator_id == "opls":
            x = self.estimator.correct(x.copy(), n_component=npc)
        return self.estimator.predict(x, n_component=npc)

    def reset_optimal_num_component(self, k) -> None:
        """
        Reset the optimal number of components for manual setup.

        Parameters
        ----------
        k: int
            Number of components according to the error plot.

        Returns
        -------
        None

        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("The number must be a positive integer.")

        if k > self._npc0:
            raise ValueError("The number must not exceed the maximum "
                             f" number of components {self._npc0}.")

        self._opt_component = k
        # re-fit the model using the updated optimal number of components
        self._create_optimal_model(self._x, self.y)

    @property
    def orthogonal_score(self) -> np.ndarray:
        """ Cross validated orthogonal score.

        Returns
        -------
        np.ndarray
            The first orthogonal scores.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._Tortho[:, self._opt_component]

    @property
    def predictive_score(self) -> np.ndarray:
        """ Cross validated predictive score.

        Returns
        -------
        np.ndarray
            The first predictive scores.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._Tpred[:, self._opt_component]

    @property
    def scores(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            The first predictive score, if the method is OPLS/OPLS-DA,
            otherwise is the scores of X

        """
        if self.estimator_id == "opls":
            return self.predictive_score
        else:
            return self.estimator.scores_x

    @property
    def q2(self) -> float:
        """ Q2

        Returns
        -------
        q2: float

        """
        return self._q2[self._opt_component]

    @property
    def optimal_component_num(self) -> int:
        """
        Number of components determined by CV.

        Returns
        -------
        int

        """
        return self._opt_component + 1

    @property
    def R2Xcorr(self) -> float:
        """
        Returns
        -------
        float
            Modeled joint X-y covariation of X.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._r2xcorr[self._opt_component]

    @property
    def R2XYO(self) -> float:
        """
        Returns
        -------
        float
            Modeled structured noise variation of X.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._r2xyo[self._opt_component]

    @property
    def R2X(self):
        """

        Returns
        -------
        float
            Modeled variation of X

        """
        return self._r2x

    @property
    def R2y(self):
        """

        Returns
        -------
        float
            Modeled variation of y

        """
        return self._r2y

    @property
    def correlation(self) -> np.ndarray:
        """ Correlation
        Returns
        -------
        np.ndarray
            Correlation loading profile

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        References
        ----------
        [1] Wiklund S, et al. Visualization of GC/TOF-MS-Based
        Metabolomics Data for Identification of Biochemically
        Interesting Compounds Using OPLS Class Models. Anal Chem.
        2008, 80, 115-122.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._corr

    @property
    def covariance(self):
        """ Covariance
        Returns
        -------
        np.ndarray
            Correlation loading profile

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        References
        ----------
        [1] Wiklund S, et al. Visualization of GC/TOF-MS-Based
        Metabolomics Data for Identification of Biochemically
        Interesting Compounds Using OPLS Class Models. Anal Chem.
        2008, 80, 115-122.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._cov

    @property
    def loadings_cv(self):
        """ Loadings from cross validation.

        Returns
        -------
        np.ndarray
            Correlation loading profile

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if self.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return np.array(self._pcv[self._opt_component+1])

    @property
    def min_nmc(self):
        """

        Returns
        -------
        float
            Minimal number of mis-classifications

        """
        return self._mis_classifications[self._opt_component]

    @property
    def mis_classifications(self):
        """

        Returns
        -------
        list
            Mis-classifications at different principal components.

        """
        return self._mis_classifications

    def _split(self, y) -> typing.Iterable:
        """
        Split total number of n samples into training and testing data.

        Parameters
        ----------
        y: np.ndarray
            Number of samples

        Returns
        -------
        iterator

        """
        n, k = y.size, self.kfold
        groups, counts = np.unique(y, return_counts=True)

        # check the number
        if counts.min() < k and n != k:
            raise ValueError(f"The fold number {k} is larger than the least"
                             f" group number {counts.min()}.")

        indices = np.arange(n, dtype=int)
        # leave one out cross validation
        if n == k:
            for i in indices:
                yield np.delete(indices, i), i

        # k fold cross validation
        else:
            group_index, blks = [], []
            for g, nk in zip(groups, counts):
                group_index.append(np.where(y == g)[0])
                blks.append(nk // k if nk % k == 0 else nk // k + 1)
            # splitting data
            for i in range(k):
                trains = np.ones(n, dtype=bool)
                for blk, idx, nk in zip(blks, group_index, counts):
                    trains[idx[blk * i: min(blk * (i + 1), nk)]] = False
                yield indices[trains], indices[np.logical_not(trains)]

    def _create_optimal_model(self, x, y) -> None:
        """
        Create final model based on the optimal number of components.
        """
        # scale data matrix
        y_scale = self.scaler.fit(y)
        x_scale = self.scaler.fit(x)

        # optimal component number
        npc = self._opt_component+1

        # fit the model
        self.estimator.fit(x_scale.copy(), y_scale.copy(), n_comp=npc)

        # summary the fitting
        self._summary_fit(x_scale, y_scale)

    def _summary_fit(self, x, y) -> None:
        """

        Parameters
        ----------
        x: np.ndarray
            scaled variable matrix.
        y: np.ndarray
            scaled dependent variable

        Returns
        -------
        CrossValidation object

        """
        npc = self._opt_component + 1
        # Calculate covariance and correlation for variable importance
        # assessment. Only works for OPLS/OPLS-DA
        if self.estimator_id == "opls":
            tp = self.estimator.predictive_score(npc)
            ss_tp = np.dot(tp, tp)
            # loadings
            w = np.dot(tp, x)
            self._cov = w / ss_tp
            self._corr = w / (np.sqrt(ss_tp) * la.norm(x, axis=0))

            # reconstruct variable matrix X
            # from orthogonal corrections.
            xrec = np.dot(self.estimator.orthogonal_scores,
                          self.estimator.orthogonal_loadings.T)
            # from predictive scores
            xrec += np.dot(
                self.estimator.predictive_score(npc)[:, np.newaxis],
                self.estimator.predictive_loadings[:, npc-1][np.newaxis, :]
            )

            # reconstruct dependent vector y
            yrec = (self.estimator.predictive_score(npc)
                    * self.estimator.weights_y[npc-1])

        else:
            xrec = np.dot(self.estimator.scores_x[:, :npc],
                          self.estimator.loadings_x[:, :npc].T)
            yrec = np.dot(self.estimator.scores_x[:, :npc],
                          self.estimator.weights_y[:npc])

        # r2x
        self._r2x = 1 - ((x - xrec) ** 2).sum() / (x ** 2).sum()
        # r2y
        self._r2y = 1 - ((y - yrec) ** 2).sum() / (y ** 2).sum()

    def _summary_cv(self) -> None:
        """
        Summary cross validation results to calculate metrics for
        assessing the model.

        Returns
        -------
        CrossValidation object

        """
        # number of mis-classifications
        _pred_class = (self._ypred > 0).astype(float)
        nmc = ((_pred_class - self.y[:, np.newaxis]) != 0).sum(axis=0)
        j = np.argmin(nmc).astype(int)
        # optimal number of components
        self._opt_component: int = j
        self._mis_classifications = nmc
        # Q2
        self._q2 = 1 - self._pressy.sum(axis=0) / self._ssy
        # metrics for OPLS
        if self.estimator_id == "opls":
            _, npc = _pred_class.shape
            # r2xcorr, r2xyo
            r2xcorr, r2xyo = [], []
            for k in range(1, npc+1):
                r2xcorr.append(
                    sum(self._ssx[k]["corr"]) / sum(self._ssx[k]["total"])
                )
                r2xyo.append(
                    sum(self._ssx[k]["xyo"]) / sum(self._ssx[k]["total"])
                )
            self._r2xcorr = r2xcorr
            self._r2xyo = r2xyo

    def _reset_y(self, y) -> np.ndarray:
        """
        Reset the labels in y to 0 and 1, and name the groups using the
        labels in y.

        Parameters
        ----------
        y: np.ndarray | list

        Returns
        -------
        np.ndarray
            Label reset in y.

        """
        if isinstance(y, list):
            y = np.array([str(v) for v in y], dtype=str)

        # groups
        labels = np.unique(y)
        # only binary classification is allowed.
        if labels.size != 2:
            raise ValueError(
                "Only binary classification is currently accepted."
            )

        # reset the values for each class
        groups = collections.defaultdict()
        y_reset = np.zeros_like(y, dtype=float)
        for i, label in enumerate(labels):
            y_reset[y == label] = i
            groups[i] = label if isinstance(label, str) else str(int(label))

        self.groups = groups
        return y_reset
