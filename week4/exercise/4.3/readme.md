## Profit Curves

In this exercise, we are going to calculate the expected value given our cost-benefit matrix for a variety of binary classifiers at different thresholds.

The data we'll be working with can be found in `data/churn.csv`.

1. Clean up the churn dataset with pandas. You should be predicting the "Churn?" column. You can drop the "State", "Area Code" and "Phone" columns as they won't be helpful features. Make sure to convert any yes/no columns to 1/0's.

2. Specify a cost-benefit matrix as a 2x2 `numpy` array. Each cell of the matrix will correspond to the corresponding cost/benefit of the outcome of a correct or incorrect classification. This matrix is domain specific, so choose something that makes sense for the churn problem. It should contain the cost of true positives, false positives, true negatives and false negatives in the following form:

    ```
    [tp   fp]
    [fn   tn]
    ```

3. Create a function `confusion_rates` that takes in a [confusion matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) (2x2 numpy array) and returns a `numpy` array of the confusion rates in the following form:

    ```
    [tpr  fpr]
    [fnr  tnr]
    ```
    
    **Note:** sklearn's confusion matrix is, well, confusing. The TN are in the upper left and the TP in the lower right. Verify this by hand by looking at the results of this code:
    
    ```python
    In [1]: from sklearn.metrics import confusion_matrix
    
    In [2]: y_true =    [1, 1, 1, 1, 1, 0, 0]
    
    In [3]: y_predict = [1, 1, 1, 1, 0, 0, 0]
    
    In [4]: confusion_matrix(y_true, y_predict)
    Out[4]:
    array([[2, 0],
           [1, 4]])
    ```
    For this example, TP = 4, TN = 2, FP = 0, FN = 1
    

4. Write a function `profit_curve` that takes these arguments:
    * `classifiers` is a list of classifier objects (like `RandomForestClassifier` and `LogisticRegression`)
    * `cb` is your cost-benefit matrix
    * `X_train`
    * `y_train`
    * `X_test`
    * `y_test`

    Here's the psuedocode for the `profit_curve` function. Note the similarity to building a ROC plot!

    ```
    function profit_curve(classifiers, cb, X_train, y_train, X_test, y_test):
    
        Calculate the class probs from the test set (2 element array with percentages
            of 1's and 0's)
            
        for each classifer in classifiers:
            Build the model on the training set
            Get the predict probabilities for the test set
            Sort the probabilities
            for each probability (in sorted order):
                Set that probability as the threshold
                Create an array of predictions of the test set where everything
                    after the threshold is True and everything before False
                Build the confusion rates matrix
                Calculate the expected profit by multipling the class probs with the
                    cost benefit matrix and the confusion rates matrix
            Plot the expected profits for the classifier
    ```

    Hints:
    * Use `predict_proba` method to get the probabilities from the classifier.
    * Use [np.argsort](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html) to sort the probabilities. This will give you the indices of the values sorted by the probabilities.
    * See if you can get the 0/1 predictions based on the threshold without using a for loop. One way is to get the indicies that should be predicted positively (they are after the threshold), create a numpy array of zeros and then set all the ones that should be predicted positively to 1.
    * Use sklearn to build the confusion matrix and use your function to get the confusion rates matrix from that.
    * In python, you can do `class.__name__` to get the name of the class. For instance `LogisticRegression.__name__` will give `"LogisticRegression"` as a string, which you can use for labeling your plots.
    * The axes of your graph:
        * x-axis: Percentage of test instances (that are predicted True)
        * y-axis: Profit

5. Run the `profit_curve` function with these models: `Logistic Regression`, `Random Forest`, and `SVM`. Don't forget to use sklearn's [StandardScalar](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to scale your data first!

6. What model and threshold yields the maximum profit? What proportion of the customer base does this target?

7. If we have a limited budget, which classifier would you choose? How about an unlimited budget?