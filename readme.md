# pypls
This package implements PLS-DA and OPLS-DA for analysis of
high-dimensional data derived from, for example, mass spectrometry
in metabolomics. The visualization of score plots, S-plot, jack-knife
confidence intervals for loading profile, and mis-classification number
in cross validation are also implemented.
## Prerequisites
This package is ceated by ```Python 3.7```, with the following packages
required:
```
numpy 1.17.2
scipy 1.3.1
matplotlib 3.1.3
```
All of these or newer version packages can be installed by using ``pip``.
## Important
This package is only workable for binary classifications. Thus if three or
more classes are in the data, this package can't handle that. An alternative
way is pair-wise classifications. As *Prof. Richard G. Brereton*
pointed out in his paper<sup>[1]</sup>, binary classification is recommended for PLS
related methods, and multi-class classification problems are not suitable
for PLS. 
## Install
The latest release can be downloaded
[**here**](https://github.com/DongElkan/pypls/releases).
Uncompress the package and set `Python` working directory there.
Since current version is not packaged, all modules must be run
under the working directory.
## Running the codes
```
# import cross validation module
import cross_validation
# import plotting functions
import plotting
``` 
1. Initialize cross validation object for 10 fold cross validation using
OPLS-DA.
    ```
    cv = cross_validation.CrossValidation(kfold=10, estimator="opls")
    ```
    Parameters:  
    `kfold`: Fold in cross validation. For leave-one-out cross validation,
    set it to `n`, is the number of samples.  
    `estimator`: The classifier, valid values are `opls` and `pls`, default is `opls`.  
    `scaler`: scaling of variable matrix.    
     * `uv`: zero mean and unit variance scaling.  
     * `pareto`: Pareto scaling. *This is the default.*  
     * `minmax`: min-max scaling so that the range for each variable is
     between 0 and 1.  
     * `mean`: zero mean scaling.  
2. Fit the model.
   ```
   cv.fit(X, labels)
   ```
   `X` is the variable matrix with size `n` (rows) by `p` (columns), where
   `n` is number of samples and `p` is number of variables.
   `labels` can be numeric values or strings, with number of
   elements equals to `n`.
3. Visualization of results.
    ```
    # construct the plotting object
    plots = plotting.Plots(cv)
    ```
    * Number of mis-classifications at different components.
    ```
    plots.plot_cv_errors()
    ```
    * Cross validated score plot.
    ```
    plots.plot_scores()
    ```
    **Note**: for OPLS-DA, predictive scores `tp` vs the first orthogonal
    scores `to` will be shown; for PLS, the first and second component will
    be shown.
    * S-plot (only suitable for OPLS-DA).
    ```
    plots.splot()
    ```
    * Loading profile with Jack-knife confidence intervals (only suitable for OPLS-DA).
    ```
    means, intervals = plots.jackknife_loading_plot(alpha=0.05)
    ```
    Where `alpha` is significance level, default is `0.05`.
    `means` are mean loadings, and `intervals` are
    Jack-knife confidence intervals.  
   
    **NOTE:** For all these plots, set `save_plot=True` and `file_name=some_string.png`
can save each plot to `some_string.png` with `dpi=1200`.
4. Model assessment.
    ```
    # R2X
    cv.R2X
    # Q2
    cv.q2
    # R2y
    cv.R2y
    # Number of mis-classifications
    cv.min_nmc
    ```
5. Access other metrics
    * Cross validated predictive scores: `cv.scores`
    * Cross validated predictive loadings: `cv.loadings_cv`
    * Optimal number of components determined by cross
    validation: `cv.optimal_component_num`
6. Prediction of new data
    ```
    predicted_scores = cv.predict(X)
    ```
    To predict the class, use
    ```
    predicted_groups = (predicted_scores >= 0).astype(int)
    ```
    This will output values of `0` and `1` to indicate the
    groups of samples submitted for prediction. `cv` object
    has the attribute `groups` storing the group names which
    were assigned in `labels` input in training. To access the
    group names after prediction, use
    ```
    print([cv.groups[g] for g in predicted_groups])
    ```
7. Other methods  
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
[2] Trygg J, Wold S. Projection on Latent Structure (OPLS). *J
Chemometr*. 2002, 16, 119-128.  
[3] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable regression
(LVR) method with a integral OSC filter. *J Chemometr*. 2003, 17, 53-64.  
[4] Wiklund S, *et al*. Visualization of GC/TOF-MS-Based Metabolomics
Data for Identification of Biochemically Interesting Compounds Using
OPLS Class Models. *Anal Chem*. 2008, 80, 115-122.
