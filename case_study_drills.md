## Model Comparison

### Prep your data
1. Download the Enron spam [dataset](http://www.aueb.gr/users/ion/data/enron-spam/). There's a script in the repo for downloading it that you can run with `./download_enron.sh`. It will take a couple minutes, but you should now have a `data` folder.

    **Note:** This is a *very large* dataset! DO NOT add it to your pull request!

2. Load the data into pandas dataframe. You can use the `build_data_frame` function from `load_enron.py`.

    ```python
    from code.load_enron import build_data_frame
    
    df = build_data_frame('data')
    ```

3. Investigate your data a little.
    * How many datapoints do you have?
    * What percent are spam and what percent are ham?

4. Create a train test split of your dataframe (70/30).

5. Use the techniques from [nlp](https://github.com/zipfian/nlp) to create a feature matrix.
    * How many features do you have?

    **Note:** Remember to figure out your features based *only on your train set*!


### kNN vs Naive Bayes

For our analysis, along with comparing accuracy and other metrics, we'd like to compare the time costs of the different algorithms. You can use ipython's `timeit` or `time`. The difference between these two is that `timeit` will run your code several times and average the time it takes (to get a more accurate measure). In standard python files, you can time your code like this:

```python
import time

start = time.time()
# YOUR CODE HERE
end = time.time()

print "total time:", end - start
```

Whenever you time something, make sure to take note of it so that you can compare the times to other models.

**NOTE:** Some of these may take a really long time to run! Use your knowledge of the algorithms and if you know something will be really slow, try running it on a small fraction of the data first!

1. Starting with the simplest classifier, k-Nearest Neighbors, we will see where it breaks down. Train the kNN classifier and time how long the training step takes.

2. Predict on the test set, again timing how long it takes. What's the accuracy?

3. Do the same training and testing with Naive Bayes. Compare the timing and accuracy results with kNN.

4. A large downside to kNN is that it is completely non-parametric. What this means is that we hardly get a decision boundary (you can construct one from the points artificially) and we get no insight into our data. With our Naive Bayes classifier, print out the top 20 words that indicate SPAM and the top 20 words that indicate HAM.

5. Any algorithm that relies on distance metrics is subject to the [curse of dimensionality](http://en.wikipedia.org/wiki/Curse_of_dimensionality). Use the UCI Machine Learning [Repository](http://archive.ics.uci.edu/ml/) to find a dataset in which kNN outperforms Naive Bayes. Also find a dataset in which Naive Bayes outperforms kNN. Some general heuristics:
    * Naive Bayes works well in the regime (n_samples << n_features)
    * kNN works well for non-linear decision boundaries
    * They both handle multi-class quite well
    * kNN usually does not perform with sparse datasets

scikit-learn has some methods for [loading](http://scikit-learn.org/stable/datasets/) data from online repositories.

**As you can see, these heuristics of when each algorithm performs best are simply that, just best guesses. Always use cross-validation to select a model and parameters.**


### Generative vs. Discriminative (Naive Bayes vs. Logistic Regression)

To get a feel for the [differences](http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf#page=14) between a Generative and Discriminative algorithm we will compare Logistic Regression to Naive Bayes.

1. First before we compare generative and discriminative algorithms, let us get a feel for  Logistic Regression on this dataset. Repeat the steps above to get the times of the train and predict as well as the accuracy for Logisitic Regression. How does it compare?

2. One area Logistic regression breaks down is in multiclass classification. kNN and Naive Bayes natively handle this case. Use the 20 Newsgroups datasets in [scikit-learn](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html) with kNN, Logistic Regression, and Naive Bayes. You will need to do a [One-vs.-all](http://scikit-learn.org/stable/modules/multiclass.html#one-vs-the-rest) classification with Logistic Regression.

3. Logistic Regression is also strictly linear (if you do not use a kernel). Try to perform a classification on a non-linear dataset. If you cannot find one, [make](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles) one!

4. Plot the decision boundary for each of the three models: kNN, Naive Bayes, Logistic Regression. [Here](http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html) is some example code. The idea of how to plot the decision boundary is to take a bunch of points over the whole surface and predict each one of them using the model.

5. Now that we have a sense of Logistic Regression, we will plot learning curves of a few difference datasets and compare them. For the following datasets, plot the error (or accuracy) of the classifier as the number of training examples increases (learning curves). A learning curve shows the accuracy of the model as a function of the size of the training set.
    * [Boston Housing Prices (predict if price > median price)](https://archive.ics.uci.edu/ml/datasets/Housing)
    * [Liver Disorders](https://archive.ics.uci.edu/ml/datasets/Liver+Disorders)
    * [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    * [Ionosphere](http://archive.ics.uci.edu/ml/datasets/Ionosphere)

6. In which regimes (# of training samples) does Naive Bayes out perform Logistic Regression (and vice versa). How do the error (or accuracy) curves compare? Does it depend on the dataset?  Are there any general trends?

### Case Study Drills

Now we'll do some rapid-fire modeling that's meant to practice your Python and scikit-learn. There are five datasets that you'll need to build a model for.

Do the following for each dataset:

1. Explore the data.
    * How many features are there?
    * How many datapoints?
    * What are feature types? (real, categorical, mixed)
    * What's the task? (What do you need to predict?)
    * What metric will you use to measure the performance of your model?

2. Do your analysis to find the best model. Try different models with different parameters and preprocessing techniques. Make sure to cross validate to get an accurate measure of performance!

3. Answer the following questions about your analysis:
    * Which modeling and preprocessing algorithms did you consider using to solve this problem?
    * What are the respective strengths/weaknesses of these tactics?
    * How did you measure performance?
    * Why did you choose your model and parameters?

4. What further steps might you take if you were to continue the project?


### The Datasets
All the datasets are from the UCI repo.

1. Predict net hourly electrical energy output (EP).
   [Dataset](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
2. Cluster patient types and briefly characterize the clusters.
   [Dataset](http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
3. Predict whether the person makes above or below $50k per year.
   [Dataset](http://archive.ics.uci.edu/ml/datasets/Adult)
4. Classify by newsgroup. [Dataset](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)
5. Generate a top 3 recommended Vroots (web sites) for each user.
   [Dataset](http://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data)
