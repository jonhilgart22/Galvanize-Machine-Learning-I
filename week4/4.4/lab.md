# Regularization Review

## Shrinkage

The regularization of the ridge is a *shrinkage*: the coefficients learned are shrunk towards zero.

The amount of regularization is set via the `alpha` parameter of the ridge,
which is tunable. The `RidgeCV` class in `scikits-learn` automatically tunes
this parameter via cross-validation.

## Exercise:
For the exercise, load the first 150 rows of the diabetes data as follows:

    ```python
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    ```

1. Fit the diabetes dataset with a Ridge Regression, use `alpha = 5` to start.
2. Now vary the values of alpha starting at zero. Plot the parameters (coefficients) of the Ridge regression (y-axis) versus the value of the alpha parameter. (There will be as many lines as there are predictors)

    ```python
    from sklearn import preprocessing
    
    k = X.shape[1]
    alphas = np.logspace(-2, 2)
    params = np.zeros((len(alphas), k))
    for i,a in enumerate(alphas):
        X_data = preprocessing.scale(X)
        fit = Ridge(alpha=a, normalize=True).fit(X_data, y)
        params[i] = fit.coef_
    
    figure(figsize=(14,6))
    for param in params.T:
        plt.plot(alphas, param)
    ```

3. Plot the test error and training error curves for Ridge regression with different alpha parameters.
   Which model would you select based on your test and training curves?



## Lasso

**The Lasso estimator** is useful for imposing sparsity on the coefficients. In
other words, it is generally preferred if we believe many of the features are
not relevant.

### Exercise:

 1. Plot the parameters (coefficients) of the LASSO regression (y-axis) versus the value of the alpha parameter.
 2. Make a plot of the training error and the testing error as a function of the alpha parameter.
 3. Select a model based on the test and training error curves.


```python
k = X.shape[1]
alphas = np.linspace(0.1, 3)
params = np.zeros((len(alphas), k))
for i,a in enumerate(alphas):
    X_data = preprocessing.scale(X)
    fit = linear_model.Lasso(alpha=a, normalize=True).fit(X_data, y)
    params[i] = fit.coef_

figure(figsize=(14,6))
for param in params.T:
    plt.plot(alphas, param)
```


### Exercise:
1.  Finally, compare three models:  your chosen Ridge model, your chosen Lasso model, and your chosen Ordinary Least Squares model.

2. What happens if you load the entire diabetes dataset instead of just the first 150 observations? This should give you some insight into cases where regularization is important.
