"""
Plot the results after cross validation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from scipy import stats
from typing import Optional, Tuple

from .cross_validation import CrossValidation


class Plots:
    """
    Plots cross validation results

    Parameters
    ----------
    cvmodel: CrossValidation object
        Cross validation model constructed in cross_validation module.

    """
    def __init__(self, cvmodel: CrossValidation):
        self._model = cvmodel

    def plot_scores(self,
                    save_plot=False,
                    file_name=None,
                    return_scores=False)\
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Plots scores. If OPLS/OPLS-DA is specified, the score plot for
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
        return_scores: bool
            Whether first and second component scores (in PLS-DA)
            or orthogonal and predictive scores (in OPLS-DA) are
            returned.

        Returns
        -------
        None

        """
        if self._model.use_opls:
            tp1 = self._model.predictive_score
            tp2 = self._model.orthogonal_score
            xlabel, ylabel = "$t_p$", "$t_o$"
        else:
            tp1 = self._model.scores[:, 0]
            tp2 = self._model.scores[:, 1]
            xlabel, ylabel = "$t_1$", "$t_2$"

        y, groups = self._model.y, self._model.groups
        # plot the figure
        _ = plt.plot(tp1[y == -1], tp2[y == -1], "o", c="r", label=groups[-1])
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
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # return scores or not
        if return_scores:
            return tp1, tp2

    def splot(self, save_plot=False, file_name=None) -> None:
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
        if not self._model.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")

        if not self._model.valid_for_splot:
            raise ValueError("S-plot only works for centered "
                             "or pareto scaled data.")

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
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def jackknife_loading_plot(self, alpha=0.05, save_plot=False,
                               file_name=None) -> tuple:
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
        # unzero mean and standard deviation
        val_ix: np.ndarray = (loading_mean == 0.) & (loading_std == 0.)
        loading_mean = loading_mean[~val_ix]
        loading_std = loading_std[~val_ix]
        # critical value
        t_critic = stats.t.ppf(1 - (alpha / 2), self._model.kfold - 1)
        # jackknife confidence interval
        loading_intervals = loading_std * t_critic
        # sort loading values
        sort_ix = loading_mean.argsort()

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
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return loading_mean, loading_intervals

    def plot_cv_errors(self, save_plot=False, file_name=None) -> None:
        """
        Plots cross validation classification errors.

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
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def vip_plot(self, xname="coef", save_plot=False, file_name=None) -> None:
        """
        Generates volcano plot-like using VIP and coefficients or
        correlation.

        Parameters
        ----------
        xname: str
            x values for x-axis of the plot, can be "coef" for
            coefficients, or "corr" for correlation.
            Defaults to "coef".
        save_plot: bool
            Whether the plot should be saved. Default is False.
        file_name
            File name for saving the plot.

        Raises
        ------
        ValueError

        """
        if not isinstance(xname, str):
            raise ValueError("Expected string for xname, got type "
                             f"{type(xname)} {xname}.")
        if xname not in ("coef", "corr"):
            raise ValueError(f"Expected 'coef' or 'corr', got {xname}.")

        if xname == "coef":
            xvals = self._model.coefcs
            xlabel = "CoeffCS[1]"
        else:
            xvals = self._model.correlation
            xlabel = "p(corr)[1]"

        if self._model.use_opls:
            ylabel = f"VIP[{self._model.optimal_component_num}+1]"
        else:
            ylabel = "VIP"

        vips = self._model.vip

        fig, ax = plt.subplots(figsize=(6, 5))
        ix = vips >= 1.
        _ = ax.plot(xvals[ix], vips[ix],
                    "o", mfc="lightcoral", mec="firebrick", ms=5., mew=1.,
                    alpha=0.8)
        _ = ax.plot(xvals[~ix], vips[~ix],
                    "o", mec="darkgray", mfc="lightgray", ms=5., alpha=0.6)
        ax.grid(visible=True, c="silver", ls="--", alpha=0.6)
        ax.set_xlabel(f"{xlabel}", fontsize=16, fontname="Times New Roman")
        ax.set_ylabel(f"{ylabel}", fontsize=16, fontname="Times New Roman")

        plt.tight_layout()

        # save the plot
        if save_plot:
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def permutation_plot(self, save_plot=False, file_name=None) -> None:
        """
        Creates permutation plot.

        Parameters
        ----------
        save_plot: bool
            Whether the plot should be saved. Default is False.
        file_name: str / None
            The name of the file to be saved.

        Returns
        -------

        """
        q2 = self._model.q2
        perm_q2 = self._model.permutation_q2
        r2 = self._model.r2
        perm_r2 = self._model.permutation_r2

        # perform the linear regression line
        x = self._model.correlation_permute_y
        n: float = float(perm_q2.size)
        tx = x.sum() / n
        # Q2
        ty = perm_q2.sum() / n
        q2_a = (q2 - ty) / (1. - tx)
        q2_b = q2 - q2_a

        # R2
        ty = perm_r2.sum() / n
        r2_a = (r2 - ty) / (1. - tx)
        r2_b = r2 - r2_a

        line_kws = {"ls": "--", "lw": 1., "alpha": 0.6, "c": "k"}
        perm_kws = {"ms": 6, "alpha": 0.6, "ls": "none", "zorder": 10.}
        metric_kws = {"marker": "^", "ms": 8, "alpha": 0.8, "ls": "none",
                      "mew": 1.5, "zorder": 10.}

        fig, ax = plt.subplots(figsize=(6, 5))
        lx = np.linspace(0., 1., num=100)
        _ = ax.plot(lx, lx * q2_a + q2_b, **line_kws)
        _ = ax.plot(lx, lx * r2_a + r2_b, **line_kws)
        _ = ax.plot([1.], [q2], mec="mediumblue", mfc="skyblue",
                    label=r"$Q^2$", **metric_kws)
        _ = ax.plot(x, perm_q2, marker="o", mec="mediumblue", mfc="skyblue",
                    label=r"Permutation $Q^2$", **perm_kws)
        _ = ax.plot([1.], [r2], mec="firebrick", mfc="lightcoral",
                    label=r"$R^2$", **metric_kws)
        _ = ax.plot(x, perm_r2, marker="s", mec="lightcoral", mfc="firebrick",
                    label=r"Permutation $R^2$", **perm_kws)
        ymin, ymax = ax.get_ylim()
        _ = ax.plot([0., 0.], [ymin, ymax], ls="-", lw=1., color="grey")
        ax.set_ylim(top=ymax, bottom=ymin)
        xmin, xmax = ax.get_xlim()
        _ = ax.plot([xmin, xmax], [0., 0.], ls="-", lw=1., color="grey")
        ax.set_xlim(left=xmin, right=xmax)
        ax.grid(visible=True, c="silver", ls="--", alpha=0.4)
        ax.set_xlabel(r"Correlation of permuted y to original y",
                      fontsize=16, fontname="Times New Roman")
        ax.set_ylabel(r"$R^2$, $Q^2$", fontsize=16)
        legs = ax.legend(loc=9, ncols=4, bbox_to_anchor=(0.5, 1.12),
                         handletextpad=0.05, columnspacing=1.2,
                         prop={'size': 12, 'family':'Times New Roman'})
        for lh in legs.legend_handles:
            lh.set_alpha(1)

        plt.tight_layout()

        # save the plot
        if save_plot:
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_permutation_dist(self, metric="q2", do_kde: bool = True,
                              save_plot=False, file_name=None) -> None:
        """
        Plots the distribution of metrics obtained from permutation test.

        Parameters
        ----------
        metric: str
            Metric used to assess the performance of the constructed model.
            "q2" and "error" are accepted as values.
            "q2": Q2
            "error": Misclassification error rate.
        do_kde: bool
            Whether to perform kernel density estimation to fit the
            distribution of permutation metrics. However, if the `error`
            is used, the kernel density estimation will not be performed.
        save_plot: bool
            Whether the plot should be saved. Default is False.
        file_name: str | None
            The name of the file to be saved.

        Returns
        -------

        """
        if metric not in ("q2", "error"):
            raise ValueError("Expected `q2`, `error`, got {}.".format(metric))

        if metric == "q2":
            metric_name = "Q2"
            mval = self._model.q2
            perm_vals = self._model.permutation_q2
            n_better = np.count_nonzero(perm_vals >= mval) + 1
        else:
            metric_name = "Error Rate"
            mval = self._model.min_nmc / self._model.y.size
            perm_vals = self._model.permutation_error
            n_better = np.count_nonzero(perm_vals <= mval) + 1
            do_kde = False

        if do_kde:
            x0 = float(int(perm_vals.min() * 10 - 1.)) / 10.
            x1 = float(int(perm_vals.max() * 10 + 1.)) / 10.
            xx = np.linspace(x0, x1, 100)
            # fit the permutation distribution
            kde = stats.gaussian_kde(perm_vals)
            dist = kde.pdf(xx)

        p = n_better / (perm_vals.size + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        _ = ax.hist(perm_vals, 100, ec="steelblue", fc="skyblue",
                    alpha=0.6, density=True)
        _ = ax.plot([mval], [0.02], marker="^", ms=8, zorder=10,
                    mec="firebrick", mfc="lightcoral", mew=1.5, clip_on=False)
        if do_kde:
            _ = ax.plot(xx, dist, "darkorange", lw=1.5)

        y0, y1 = ax.get_ylim()
        ax.plot([mval, mval], [y0, y1], "--",
                c="lightcoral", lw=0.8, alpha=0.6)
        ax.set_ylim((y0, y1))

        ax.set_xlabel(metric_name, fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.set_title(f"Permutation {metric_name} Distribution", fontsize=16)

        pstr = "%.2e" % p if p < 0.01 else "%.4f" % p
        mstr = (f"{mval:.2f}" if abs(mval) >= 0.01 or mval == 0.
                else '%.2e' % mval)
        labels = [f"{metric_name} = {mstr}", f"$p = ${pstr}"]
        h = mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                  lw=0, alpha=0)
        ax.legend([h, h], labels, loc='best',
                  prop={'family': "Times New Roman", 'size': 12},
                  fancybox=True, framealpha=0.6, edgecolor="darkred",
                  facecolor="snow", handlelength=0, handletextpad=0)

        plt.tight_layout()

        # save the plot
        if save_plot:
            if not file_name.endswith(".png"):
                file_name += ".png"
            plt.savefig(file_name, dpi=1200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
