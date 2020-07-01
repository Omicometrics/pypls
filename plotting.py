"""
Plot the results after cross validation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Plots:
    """
    Plot cross validation results

    Parameters
    ----------
    cvmodel: CrossValidation object
        Cross validation model constructed in cross_validation module.

    """
    def __init__(self, cvmodel):
        self._model = cvmodel

    def plot_scores(self, save_plot: bool = False,
                    file_name: str = None) -> None:
        """
        Plot scores. If OPLS/OPLS-DA is specified, the score plot for
        OPLS/OPLS-DA is used, i.e., the first component of orthogonal
        versus predictive scores are used for the plot, otherwise, the
        first two components of score plots are used.

        Parameters
        ----------
        save_plot: bool
            Whether the plot should be saved. Default is False.
        file_name: str | None
            File name for saving the plot. They should be compatible
            in Matplotlib. The figure format supported by Matplotlib
            can be found at
            https://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.get_supported_filetypes
            Default is "png". If the file_name doesn't have dot, an
            extension of "png" will be added, but if the string after
            the dot is not supported by Matplotlib, an error will be
            raised. Thus if the extension is not intended to be
            specified, dot shouldn't be present in file_name.

        Returns
        -------
        None

        """
        if self._model.estimator_id == "opls":
            tp1 = self._model.predictive_score
            tp2 = self._model.orthogonal_score
            xlabel, ylabel = "$t_p$", "$t_o$"
        else:
            tp1 = self._model.scores[:, 0]
            tp2 = self._model.scores[:, 1]
            xlabel, ylabel = "$t_1$", "$t_2$"

        y, groups = self._model.y, self._model.groups
        # plot the figure
        _ = plt.plot(tp1[y == 0], tp2[y == 0], "o", c="r", label=groups[0])
        _ = plt.plot(tp1[y == 1], tp2[y == 1], "^", c="b", label=groups[1])
        # set up axis limits
        xlim, ylim = plt.xlim(), plt.ylim()
        _ = plt.plot(xlim, [0, 0], "k--", lw=0.8)
        _ = plt.plot([0, 0], ylim, "k--", lw=0.8)
        _ = plt.xlim(xlim)
        _ = plt.ylim(ylim)
        _ = plt.xlabel(xlabel, fontsize=16)
        _ = plt.ylabel(ylabel, fontsize=16)
        _ = plt.legend(frameon=False, loc="upper right",
                       bbox_to_anchor=(1, 1.1), ncol=2, fontsize=12)
        plt.tight_layout()

        # save plot
        if save_plot:
            if "." not in file_name:
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")

        plt.show()

    def splot(self, save_plot: bool = False,
              file_name: bool = None) -> None:
        """
        S-plot

        Parameters
        ----------
        save_plot: bool
            Whether the plot should be saved. Default is False.
        file_name: str | None
            File name for saving the plot. They should be compatible
            in Matplotlib. The figure format supported by Matplotlib
            can be found at
            https://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.get_supported_filetypes
            Default is "png". If the file_name doesn't have dot, an
            extension of "png" will be added, but if the string after
            the dot is not supported by Matplotlib, an error will be
            raised. Thus if the extension is not intended to be
            specified, dot shouldn't be present in file_name.

        Returns
        -------
        None

        References
        ----------
        [1] Wiklund S, et al. Visualization of GC/TOF-MS-Based
        Metabolomics Data for Identification of Biochemically
        Interesting Compounds Using OPLS Class Models. Anal Chem.
        2008, 80, 115-122.

        """
        if self._model.estimator_id != "opls":
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")

        # covariance and correlations
        covx = self._model.covariance
        corrx = self._model.correlation

        # plot
        fig, ax = plt.subplots(figsize=(10, 5))
        _ = ax.scatter(
            covx, corrx,
            marker="o", s=40, c=covx, cmap="jet", edgecolors="none"
        )
        _ = ax.set_xlabel("cov($t_p$, X)", fontsize=16)
        _ = ax.set_ylabel("corr($t_p$, X)", fontsize=16)
        plt.colorbar(ax.get_children()[0], ax=ax)

        # save plot
        if save_plot:
            if "." not in file_name:
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")

        plt.show()

    def jackknife_loading_plot(self, alpha: float = 0.05,
                               save_plot: bool = False,
                               file_name: str = None) -> tuple:
        """
        Loading plot with Jack-knife intervals.

        Parameters
        ----------
        alpha: float
            Significance level for calculating the intervals.
            Default is 0.05.
        save_plot: bool
            Whether the plot should be saved. Default is False.
        file_name: str | None
            File name for saving the plot. They should be compatible
            in Matplotlib. The figure format supported by Matplotlib
            can be found at
            https://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.get_supported_filetypes
            Default is "png". If the file_name doesn't have dot, an
            extension of "png" will be added, but if the string after
            the dot is not supported by Matplotlib, an error will be
            raised. Thus if the extension is not intended to be
            specified, dot shouldn't be present in file_name.

        Returns
        -------
        loading_mean: np.ndarray
            Mean of cross validated loadings.
        loading_interval: np.ndarray
            Jack-knife confidence intervals

        """
        # mean loadings
        loading_mean = self._model.loadings_cv.mean(axis=0)
        loading_std = self._model.loadings_cv.std(axis=0)
        # critical value
        t_critic = stats.t.ppf(1 - (alpha / 2), self._model.kfold - 1)
        # jackknife confidence interval
        loading_intervals = loading_std * t_critic
        # sort loading values
        sort_ix = np.argsort(loading_mean)

        # plot with bar error
        errorbar_fmt = {"linewidth": 0.8, "linestyle": "-"}
        bar_x = np.arange(loading_mean.size)
        fig, ax = plt.subplots(figsize=(10, 5))
        _ = ax.bar(
            bar_x, loading_mean[sort_ix], yerr=loading_intervals[sort_ix],
            width=1, capsize=2, error_kw=errorbar_fmt, color="none",
            edgecolor="cornflowerblue"
        )
        _ = ax.set_xlim(left=-0.5, right=loading_mean.size + 0.5)
        xlim = ax.get_xlim()
        _ = ax.plot(xlim, [0, 0], "k", linewidth=0.6)
        _ = ax.set_xlim(xlim)
        _ = ax.set_xlabel("Variable", fontsize=16)
        _ = ax.set_ylabel("cov($t_p$, X)", fontsize=16)
        plt.tight_layout()

        # save the plot
        if save_plot:
            if "." not in file_name:
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")

        plt.show()

        return loading_mean, loading_intervals

    def plot_cv_errors(self, save_plot: bool = False,
                       file_name: str = None) -> None:
        """ Plot cross validation classification errors.

        Returns
        -------
        None

        """
        nmc = self._model.mis_classifications
        _ = plt.plot(np.arange(len(nmc)) + 1, nmc,
                     marker="o", mfc="none", markersize=5.)
        _ = plt.xlabel("Number of Components", fontsize=16)
        _ = plt.ylabel("Number of Misclassifications", fontsize=16)
        _ = plt.xlim(left=0)
        plt.tight_layout()

        if save_plot:
            if "." not in file_name:
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")

        plt.show()
