# Introduction to Machine Learning

In this exercise we will bridge the gap between what you learned in the first semester and the Machine Learning of this class.  We will compare a statistical approach (Normal Equation) with a computational (Gradient Descent) machine learning method to solve for the coefficients of OLS Regression.

This lab will also serve as our introduction to `scikit-learn`, a very powerful machine learning library for Python that we will leverage throughout this course.

## Goals

* In what situation should you use Gradient Descent with OLS?
* In what situation is it preferable to use The Normal Equation?
* What is the hypothesis function for Linear Regression?
* What is the typical cost function for linear regression?
    * Can you name 1 other cost function an a scenario in which it is preferred?

####References:####
* [Andrew Ng's Machine Learning Lecture Notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf)

####Notes on Implementing Gradient Descent:####
* Implementing gradient descent can lead to challenging debugging. Try
computing values by hand for a really simple example (1 feature, 2 data points)
and make sure that your methods are getting the same values.
* Even though the first data set we are giving you only has one feature, make
sure your code is general and can work with matrices of any size.
* Numpy is your friend. Use the power of it! There should only be one loop in
your code (in `run`). You should never have to loop over a numpy array. See the
numpy [tutorial](http://wiki.scipy.org/Tentative_NumPy_Tutorial) and
[documentation](http://docs.scipy.org/doc/).

## Data

We're going to be working with a few datasets today. They all have inherent
prediction problems to solve.

1. `food_trucks.csv`: data on the profit for a foodtruck and the population
density of the city they are in. The goal is to predict the profit based on the
population density.
2. `housing_prices.csv`: data on the price of home sales in Portland, Oregon.
There are two other variables: square footage of the home and number of rooms.
The goal is to predict the price.
3. `bikes.csv`: data on the number of Capital Bikeshare bikes used on a given
day in Washington DC.
`bikes.txt` contains an explanation of all of the columns.

## Iteration Zero: Explore the data

For implementing gradient descent, we are going to start simple
data set: `food_trucks.csv`. It will be easier to verify that your
implementation is correct with this data set. It also only has 2 variables, so
we can visualize it in two dimensions (most data has way too many dimensions to
visualize).

1. Make a scatter plot with the population on the x axis and the profit on the
y axis.

2. Just by eye-balling, make an estimate for a good coefficient for this
equation: `y = mx`. We are using this form since our first implementation will
not find an intercept. But also make an estimate for good coefficients for this
equation: `y = mx+b`.

## Iteration One: Hypothesis Function

It may seem slightly trivial in the simple case of linear regression to explicitly define a hypothesis function, but by abstracting the hypothesis function we can write a general Gradient Descent algorithm that we can easily plug any other hypothesis function into (for example a Logistic function).

Hopefully you can start to see (even this early on) the power of generalizing a machine learning model into these three components:

1. Hypothesis
2. Cost
3. Optimization

By doing so our analysis can be much more flexible.  Recall that for OLS the hypothesis function we will be using is the following:

![](images/hypo.png)

1. Fill in the `hypothesis()` function to correctly represent the OLS linear hypothesis.  It should make no assumptions about the number of features or shape of data.

## Iteration Two: Cost Function

In order to be able to evaluate if our gradient descent algorithm is working
correctly, we will need to be able to calculate the cost.

The cost function we will be using is the Residual Sum of Squares. We will also
be implementing R^2 (this allows us to compare our results with those of the
sklearn LinearRegression).

As a reminder, the formulas are below. *y* is the true value and *f* is the
predicted value. *y* with the bar is the mean. RSS is the same as SSres.

![rss](images/rss.png)

![sstot](images/sstot.png)

![r2](images/r2.png)

1. Implement `compute_cost` and `compute_r2` (in `ordinary_least_squares.py`).  Test this manually with a dummy array that you can compute the cost of by hand.

## Iteration Three: Implement Gradient Descent

Now are are going to implement gradient descent, an algorithm for solving
optimization problems.

Below is psuedocode for the gradient descent algorithm. This is a generic
algorithm that can solve a plethora of optimization problems.
In our case, the *x* we are solving for is the coefficient vector *b* in
*y = Xb*. We will initialize it to be all zeros.

In this pseudocode and in our implementation, we will stop after a given number
of iterations. Another valid approach is to stop once the incremental
improvement in the cost function is sufficiently small.

    Gradient Descent:
        input: J: differential function (cost function)
               x0: initial solution
               alpha: learning rate
               n: number of iterations
        output: x: local minimum of cost function J

        x <= x0
        repeat for n iterations:
            x <= x - alpha * gradient(J)

You are going to be completing the code stubs in the following cells.

1. In order to initialize our GradientDescent, we should tell it which model/hypothesis we will be using.  In this case create an instance of `OrdinaryLeastSquares` and pass it to the initializer of `GradientDescent`.

1. Implement the methods `cost` and `score`. These aren't directly needed for
the gradient descent algorithm, but it's really helpful for debugging to output
the cost after every iteration to verify that it is increasing and also to
evaluate the success of your model. Use the coeffs instance variable
to calculate. Call the `compute_cost` and `compute_r2` functions.

2. In every iteration of gradient descent, we need to update the coefficients
to improve the cost function. We do this by moving the coefficients in the
direction of decreasing gradient (*gradient descent*).
Recall that the following is the generic
update method. *J* is the cost function and *theta* is the coefficient vector.

    ![update](images/update.png)

    In order to update our coefficients, we need to compute the gradient of the
    cost function. In our case, the cost function is RSS, the gradient of which
    is as follows. *h(x)* is the evaluation of the prediction of *x* with the
    current coefficents.

    ![gradient](images/gradient.png)

    Implement the `gradient` method.

3. We are almost there, now all that remains is to write the scaffold code to
repeatedly run the update function until convergence. Write the `run` method,
which iteratively updates the coefficients instance variable using the
`gradient` method and the learning rate `alpha`.

4. Fill in the `predict` method so that you can see the predicted values on
new data.

## Iteration Three: Run gradient descent & compare

Now we're ready to try out our gradient descent algorithm on some real data.

1. In `food_trucks.py`, load in the data with pandas, and convert it to numpy
arrays. Use `train_test_split` to break the data in training and testing data.

2. Import your `GradientDescent` class and try running it:

    ```python
    gd = GradientDescent()
    gd.run(X_train, y_train)
    print "coeffs:", gd.coeffs
    print "R^2:", gd.score(X_test, y_test)
    ```

    Are they similar to your estimate in Part A?

    **Note:** If you're having trouble getting it to converge, run it for just
    a few iterations and print out the cost at each iteration. The value should
    be going down. If it isn't, you might need to decrease your learning rate.
    And of course check your implementation to make sure it's correct. You can
    also try printing out the cost every 100 iterations if you want to run it
    longer and not get an insane amount of printing.

3. As you can see, in all these cases, we only get one coefficient. Ideally we
would like to also have an intercept. In the one feature case, our equation
should look like this: `y = mx + b` (not just `y = mx`). We solve this by adding
a column of ones to our feature matrix. Implement `add_intercept` in
`linear_regression_functions.py` and use it to modify your feature matrix
before running gradient descent or using sklearn's linear regression.

    Modify the `__init__` method of `GradientDescent` so that it can take a boolean parameter `fit_intercept`:
    
    ```python
    def __init__(self, fit_intercept=True):
        # code goes here
    ```
    
    If you set `add_intercept` to be False, it should work the same way as before this modification.
    
    It might be helpful to add this function in `linear_regression_functions.py`:
    
    ```python
    def add_intercept(X):
    '''
    INPUT: 2 dimensional numpy array
    OUTPUT: 2 dimensional numpy array

    Return a new 2d array with a column of ones added as the first
    column of X.
    '''
    ```

## Extra Credit: Scaling

If you try running your gradient descent code on the bike data, or even the
housing data, you'll probably have issues with it converging. You can try
playing around with alpha, the learning rate (one option is to decrease the
learning rate at every iteration).

An easier way is to *scale* the data.
Basically, we shift the data so that the mean is 0 and the standard deviation
is 1. To do this, we compute the mean and standard deviation for each feature
in the data set and then update the feature matrix by subtracting each value
by the mean and then dividing by the standard deviation.

1. Commit your code! Run a `git commit -m "some message"` so that if you goof
things up with your changes you don't lose your previous version.

2. Add the some methods to the `GradientDescent` class to calculate the scale factors and to scale the features (if the parameter is set.
    * **Note:** Make sure to scale before you add the intercept column. You don't want to try and scale a column of all ones.

3. Modify the `__init__` method of the `GradientDescent` class so that it can take a boolean `scale` parameter. Add calls of the above functions to the `run` method if `scale` is `True`.

4. Make sure `cost`, `score` and `predict` scale their feature matrices.
    * **Note:** You should calculate mu and sigma from the *training* data and use those values of mu and sigma to scale your test data (that would result in dividing by 0).

5. Try running your code on the housing data. Does it converge? Do you get
similar coefficients and R^2 as sklearn? Finally try the bikes data.

### [Advanced Algorithms](http://imgur.com/a/Hqolp)

![](http://i.imgur.com/2dKCQHh.gif?1)

![](http://i.imgur.com/pD0hWu5.gif?1)

![](http://i.imgur.com/NKsFHJb.gif?1)