## k Nearest Neighbors
This algorithm is very simple to implement. Note that it takes nothing to train the model, you just need to save the 
data. When given a new data point, you need to calculate the distance of that data point to every existing data point
and find the *k* closest ones.

### Data

You can also use sklearn's `make_classification` for creating a fake dataset for this exercise:

```python
X, y = make_classification(n_features=4, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, class_sep=5, random_state=5)
```

## kNN Implementation

Include all code for this section in the file `KNearestNeighbors.py`
Here's the pseudocode for `k` Nearest Neighbors:

    kNN:
        for every point in the dataset:
            calculate the distance between the point and x
            sort the distances in increasing order
            take the k items with the smallest distances to x
            return the majority class among these items

1. Implement the function `euclidean_distance` which computes the Euclidean distance between two numpy arrays. Euclidean distance only works for continuous features.

2. Implement the class `KNearestNeighbors`. We are going to write our code similar to how sklearn does. So you should be able to run your code like this:

    ```python
    knn = KNearestNeighbors(k=3, distance=euclidean_distance)
    knn.fit(X, y)
    y_predict = knn.predict(X)
    ```

    Here `X` is the feature matrix as a 2d numpy array, `y` is the labels as a numpy array. 3 is the *k* and `euclidean_distance` is the distance function. `predict` will return a numpy array of the predicted labels.

    You will need to implement a `KNearestNeighbors` class with three methods: `fit`, `predict` and `score` (calculates accuracy).

3. Implement `cosine_distance` which computes the cosine distance function. This gives the angle between the two vectors. Here is the formula:

    ![cosine distance](images/cosine.png)

4. Plot the decision boundary. Look at this [sklearn example](http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py). Note that you'll need exactly 2 continuous features in order to do this.

5. Test your algorithm on a dataset used for a previous exercise. Use [sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) to compute the accuracy, precision and recall of your model. Use KFold Cross Validation and determine the best choice of `k` (will probably depend on which metric you use!).

