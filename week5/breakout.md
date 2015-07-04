# Kernel functions

Kernel functions are a very powerful abstraction that decouples the feature space of the data from the data itself.  What does this mean? Kernels (similarly, dimensionality reduction) transform the initial feature space.  But rather than explicitly change the feature matrix, kernels implicitly apply their transformation anytime it encounters the initial feature in the machine learning algorithm.

## Kernels Applied

Before getting into kernels applied to Support Vector Machines (SVMs), we'll familiarize ourselves with just the concept.  Kernels are general enough (and powerful enough) that they can be applied to a wide variety of machine learning methods.

### XOR

The exclusive OR is a charateristically difficult task for many machine learning algorithms (especially linear ones).  Kernels are powerful enough to fit a complex decision boundary, but remain simple and somewhat interpretable as a model.

![](images/xor.png)

1. Use numpy to generate some fake data:

 ```python
 np.random.seed(0)
 X = np.random.randn(300, 2)
 Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
 plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
 ```

 _NOTE: When exploring a new method, try to find the simplest *non-trivial* data to test the model.  Often it is easier to generate this data_

 #### Logistic Regression

2. To get a healthy baseline, use scikit-learn's `LogisticRegression` to fit a basic model to the data.

3. Perform `k`-fold cross validation to evaluate the goodness of the model.  What is the accuracy, precision, and recall on the `XOR` data?

 These point estimates (accuracy, precision, recall) only tell half of the story.  It is often very illuminating to plot the decision boundary of your model to "debug" the misclassifications.

4. Using `matplotlib`, plot the initial data (as shown above) with the decision boundary of the `LogisticRegression` overlayed. [Here](http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html) is an example code for plotting decision boundary.

 As you can see, the logistic regression model is not complex enough to describe this highly non-linear boundary.  To start and really understand kernels, we will generate functions ourselves.

 #### Polynomial Kernel

5. The goal is to implement the polynomial kernel function to the data:

 ![](images/kernel_func.gif)

 Using this function, create _Kernel matrix_:

 ![](images/kernel_mat.gif)

 The simplest way to do this is to first create a symmetric matrix from your feature matrix: `X.dot(X.T)`. Every element in this matrix is the dot product of two row vectors of the original `X` matrix, i.e., `X.XT = [xi^T xj]`. This is a _n x n_ matrix where _n_ is total number of rows in the original data matrix.  
Now apply the kernel funciton to every element of `X.XT`. To do this we can use the `numpy.vectorize` function.

6. Refit the logistic regression on this new kernel matrix. In this case, we can treat every row in the kernel matrix as a new data point. Plot the decision boundary. Does it perform better?

7. Repeat the above process for different values of _d_ in the kernel function. For the XOR data, the quadratic function performs really well with logistic regression.


 #### Extra: RBF Kernel

 The polynomial kernel may or may not have done better than simply no kernel at all.  Just like with any hyperparameters, it could take some tuning to get the best fit with a kernel.

 Rather than tweaking the polynomial kernel to fit this data, let us try one of the most used kernels: the Radial Basis Function

 The RBF kernel is one of the most used kernels because it is highly generalizable and can create arbitrarily complicated decision boundaries (but care should be taken to not overfit).

8. Use scipy's pairwise distances function with a Gaussian distance function to compute the RBF kernel.  Use the 4 exemplar points as your landmarks: (0,0) ; (0,1) ; (1, 0) ; (1, 1).

9. Similar to above, use a `meshgrid` to plot the decision boundary of the RBF kernel.
10. Can you think of/generate data in which a RBF kernel performs much better than a polynomial kernel?


