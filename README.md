# pypls
This package implements PLS-DA and OPLS-DA for analysis of
high-dimensional data derived from, for example, mass spectrometry
in metabolomics. The visualization of score plots, S-plot, jack-knife
confidence intervals for loading profile, and mis-classification number
in cross validation are also implemented.
# Important Update
Starting from ```version 1.0.0```, heavy calculations, *e.g.*, NIPALS of PLS/OPLS, 
calculations of scores and loadings *etc.*, are implemented by **[Cython](https://github.com/cython/cython)**.
This implementation can significantly speed up the calculations, especially for
permutation tests.
## Prerequisites
This package is created by ```Python 3.7```, and recently tested by `Python 3.10`, with the following packages
required:
```
numpy 1.17.2
scipy 1.3.1
matplotlib 3.1.3
tqdm 4.64.0
Cython 3.0
```
All of these or newer version packages can be installed by using ``pip``.
## Important Note
This package is only workable for binary classifications. Thus, if three or
more classes are in the data, this package can't handle that. An alternative
way is pair-wise classifications. As *Prof. Richard G. Brereton*
pointed out in his paper<sup>[1]</sup>, binary classification is recommended for PLS
related methods, and multi-class classification problems are not suitable
for PLS. 
## Install
The latest release can be downloaded
[**here**](https://github.com/DongElkan/pypls/releases).
Uncompress the package. In command line, *e.g.*, ```Command Prompt``` in Windows,
```terminal``` in iOS, run
```
python -m setup install
```
This will compile all ``cython`` codes on the OS, install the package, and run the package using any tool.
## Run `pypls`
```
# import cross validation module
from pypls import cross_validation
# import plotting functions
from pypls import plotting
``` 
* ##### Initialize cross validation object for 10-fold cross validation using OPLS-DA.
    ```
    cv = cross_validation.CrossValidation(kfold=10, estimator="opls")
    ```
    Parameters:  
    `kfold`: Fold in cross validation. For leave-one-out cross validation,
    set it to `n`, the number of samples.  
    `estimator`: The classifier, valid values are `opls` and `pls`. Defaults to `opls`.  
    `scaler`: scaling of variable matrix.    
     * `uv`: zero mean and unit variance scaling.  
     * `pareto`: Pareto scaling. *This is the default.*  
     * `minmax`: min-max scaling so that the range for each variable is
     between 0 and 1.  
     * `mean`: zero mean scaling.
* ##### Fit the model.
   ```
   cv.fit(X, labels)
   ```
   `X` is the variable matrix with size `n` (rows) by `p` (columns), where
   `n` is number of samples and `p` is number of variables.
   `labels` can be numeric values or strings, with number of
   elements equals to `n`.
* ##### Permutation test <sup>[5, 6]</sup>    
    To identify whether the constructed model is overfitting, permutation
test is generally applied, by repeatedly simply randomizing labels and performing
the model construction and prediction on the randomized labels many times. This
package adopts same strategy, which uses
    ```
    cv.permutation_test()
    ```
    Parameters:  
    `num_perms`: Number of permutations. Defaults to `10000`.  
    To get _p_ value, the significance of the constructed model, run
    ```
   cv.p(metric="q2")
   ```
   Parameters:  
   `"q2"`: Q2.  
   `"error"`: Mis-classification error rate.
   > **IMPORTANT**  
   > _p_ value is calculated as <sup>[7]</sup>  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_p_ = (No. of permutation error rate <= normal error rate + 1) / (n + 1)  
    if misclassification rate (_i.e._, parameter `error`) is used as the metric, or  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_p_ = (No. of permutation Q2 >= normal Q2 + 1) / (n + 1)  
    if Q2 (_i.e._, parameter `q2`) is used, and `n` is the number of permutations.
* ##### Visualization of results.
    ```
    # construct the plotting object
    plots = plotting.Plots(cv)
    ```
  * Number of mis-classifications at different principal components:
    ```
    plots.plot_cv_errors()
    ```
  * Cross validated score plot:
    ```
    plots.plot_scores()
    ```
    > **NOTE**  
    For OPLS-DA, predictive scores `tp` vs the first orthogonal scores `to`
    will be shown; for PLS, the first and second component will be shown.
  * S-plot (only suitable for OPLS-DA).
    ```
    plots.splot()
    ```
  * Loading profile with Jack-knife confidence intervals (only suitable for OPLS-DA).
    ```
    means, intervals = plots.jackknife_loading_plot(alpha=0.05)
    ```
    Where `alpha` is significance level, defaults to `0.05`.
    `means` are mean loadings, and `intervals` are
    Jack-knife confidence intervals.  
  * VIP<sup>[9, 10]</sup> plot
    ```
    cv.vip_plot(xname="coef")
    ```
    This will show coefficients obtained from regression of centered and scaled
    data matrix `X` to scaled `y` and against VIP (variable importance/influence
    on projection), corresponding to *CoeffCS vs VIP* in SIMCA.
    Parameters:
    * `xname`: the *x data vs* VIPs. Valid values are:
      * `"coef"`: CoeffCS, centered and scaled coefficients.
      * `"corr"`: Correlation scaled loading, *i.e.*, `p(corr)`.
    > **NOTE**  
    For OPLS-DA, the VIP<sub>4,tot</sub> of ref [10](#references) is used as 
    the VIP values of the plot.
  * Permutation plot<sup>[8]</sup>
    ```
    plots.permutation_test()
    ```
    
    This will generate a permutation test plot, _Correlation of permuted y to 
    original y_ vs _R2_ and _Q2_, with a regression line.
  * Distribution of the metrics obtained from permutation test
    ```
    plots.plot_permutation_dist(metric="q2")
    ```
    Parameters:
    * `metrics`: the metric used to show the permutation test results. Valid
      values are:
      * `"q2"`: Q2.
      * `"error"`: Mis-classification error rate.
    * `do_kde`: Whether to fit the distribution using kernel density estimation (KDE).
    `True` for yes, `False` for not to fit the distribution. Note that, if `error`
    is used to show the distribution, KDE will not be used as the error rate is
    not continuous, thus the estimation will be inaccurate.
    
   > **NOTE**  
   > For all above plots, set `save_plot=True` and `file_name=some_string.png`
   can save each plot to `some_string.png` with `dpi=1200`.
* ##### Model assessment.
    ```
    # R2X
    cv.R2X_cum
    # Q2
    cv.q2
    # R2y
    cv.R2y_cum
    # Number of mis-classifications
    cv.min_nmc
    ```
   To check the `R2X` and `R2y` of the optimal component, _i.e._,
`cv.optimal_component_num`, call `cv.R2X` and `cv.R2y`.
* ##### Access other metrics.
    * Cross validated predictive scores: `cv.scores`
    * Cross validated predictive loadings: `cv.loadings_cv`
    * Optimal number of components determined by cross validation: `cv.optimal_component_num`
    * VIP values: `cv.vip`
    * Index of variables used for the analysis: `cv.used_variable_index`
    > **NOTE**  
    In the analysis, the variable that has a single value, which happens when 
    all values are missing, is excluded. Therefore, to map the values of, *e.g.*, 
    `cv.loadings_cv` and `cv.vip` to corresponding variable names, the index of
    variables must be output so that, taking VIP values for instance,
    `cv.vip[i]` is the VIP value of <ins>`cv.used_variable_index[i]`th</ins> variable.
* ##### Prediction of new data.
    ```
    predicted_scores = cv.predict(X, return_scores=False)
    ```
    To predict the class, use
    ```
    predicted_groups = (predicted_scores >= 0).astype(int)
    ```
    This will output values of `0` and `1` to indicate the
    groups of samples submitted for prediction. `cv` object
    has the attribute `groups` storing the group names which
    were assigned in `labels` input for training. To access the
    group names after prediction, use
    ```
    print([cv.groups[g] for g in predicted_groups])
    ```
    Set `return_scores=True` will return predictive scores for OPLS.
* ##### Other methods.  
    `cv` provides a method `reset_optimal_num_component` to reset
    the optimal number of components manually, instead of defaultedly
    at the minimal number of mis-classification.
    ```
    cv.reset_optimal_num_component(n)
    ```

## Author
Nai-ping Dong  
Email: naiping.dong@hotmail.com

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/DongElkan/pypls/blob/master/LICENSE) for details.

## References
[1] Brereton RG, Lloyd GR. Partial least squares discriminant analysis:
taking the magic away. *J Chemometr*. 2014, 18, 213-225.
[Link](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2609)  
[2] Trygg J, Wold S. Projection on Latent Structure (O-PLS). *J
Chemometr*. 2002, 16, 119-128.
[Link](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.695)   
[3] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable regression
(LVR) method with a integral OSC filter. *J Chemometr*. 2003, 17, 53-64.
[Link](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.775)  
[4] Wiklund S, *et al*. Visualization of GC/TOF-MS-Based Metabolomics
Data for Identification of Biochemically Interesting Compounds Using
OPLS Class Models. *Anal Chem*. 2008, 80, 115-122.
[Link](https://pubs.acs.org/doi/abs/10.1021/ac0713510)  
[5] Bijlsma S, *et al*. Large-Scale Human Metabolomics Studies: A Strategy for
Data (Pre-) Processing and Validation. *Anal Chem*. 2006, 78, 2, 567–574.
[Link](https://pubs.acs.org/doi/10.1021/ac051495j)  
[6] Ojala M, *et al*. Permutation Tests for Studying Classifier Performance.
*J Mach Learn Res*. 2010, 11, 1833−1863.
[Link](https://www.jmlr.org/papers/v11/ojala10a.html)  
[7] North BV, *et al*. A Note on the Calculation of Empirical P Values from
Monte Carlo Procedures. *Am J Hum Genet.* 2002, 71(2), 439–441. [Link](https://www.jmlr.org/papers/v11/ojala10a.html)  
[8] Lindgren F, *et al*. Model validation by permutation tests: Applications
 to variable selection. *J Chemometrics*. 1996, 10, 521-532. [Link](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/%28SICI%291099-128X%28199609%2910%3A5/6%3C521%3A%3AAID-CEM448%3E3.0.CO%3B2-J)  
[9] Wold S, *et al*. PLS—partial least-squares projections to latent structures.
In 3D QSAR in Drug Design, Theory Methods and Applications, H Kubinyi (eds.). 
ESCOM Science Publishers: Leiden, 1993, 523–550.
[10] Galindo-Prieto B, *et al*. Variable influence on projection (VIP) for
orthogonal projections to latent structures (OPLS). *J Chemometrics*. 2014,
28(8), 623-632. [Link](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.2627)

