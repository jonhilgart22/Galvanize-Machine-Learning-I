## Other Bayes

Now that you have seen how Multinomial Naive Bayes is implemented, we are going to better solidify our knowledge of the Naive Bayes family of variants and implement [Bernoulli Bayes](http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html) as well as [Gaussian Bayes](http://www.autonlab.org/tutorials/gaussbc12.pdf)

### Implementation

To implement these new Naive Bayes variants, we will actually be utilizing much of the framework we used in the in-class lab.  The reason is twofold:

* Using OO programming we can minimize the additional code we need and use proper software abstractions.
* The unification of the multiple variants will be much more obvious.

The second point here is one that should not be understated.  And as seen in lecture, we can actually use **any distribution** to model each feature.  This makes Naive Bayes a flexible and powerful technique.

_For each of these methods we only need to adapt the likelihood function and potentially the predict function._

## Bernoulli Bayes

The only difference between each of these variants and Multinomial Bayes is in the distribution of likelihoods.  Recall that for Bernoulli Bayes we have the following function:

![](../images/bernoilli_bayes.png)

1. Inherit from the `NaiveBayes` class from the lab to subclass a `BernoulliBayes` class.

2. Overwrite the `_compute_likelihood()` function of the `NaiveBayes` parent class to reflect the likelihood of the Bernoulli Bayes variant.

3. Run your `BernoulliBayes` model on the same spam dataset from lab.  How does it score compared to the Multinomial Naive Bayes?

  ## Gaussian Bayes

  ![](../images/gaussian_bayes.png)

4. Create a `GaussianBayes` subclass and overwrite the `_compute_likelihood()` function of the `NaiveBayes` parent class to reflect the likelihood of the Gaussian Bayes variant.

5. Train your `GaussianBayes` model on the same spam dataset from lab.  How does it score compared to the Multinomial Naive Bayes?

  Gaussian Bayes is typically much better suited to continuous features (since the Gaussian PDF is continuous).

6. Train your `GaussianBayes` model this time on the `cars_scrubbed.csv` dataset to predict the `origin` of the car.  How well does it score?

## Extra Credit: Out of Core learning

Naive Bayes is an interesting an unique classifier in that it can scale to large amounts of training data.  Part of this is inherent in the algorithm itself since it usually only requires counts of the input data.  And part of this stems from the fact that Naive Bayes is an [online algorithm](http://en.wikipedia.org/wiki/Online_machine_learning) that can incrementally be trained.  This allows you to stream data to the model incrementally to train and alleviates any memory constraints (and makes it well suited to the problem of spam email).

1. Implement a `partial_fit()` method on your `NaiveBayes` parent class that allows you to train on individual examples sequentially.

2. Train this classifier on the entire [Wikipedia corpus](http://www.cs.upc.edu/~nlp/wikicorpus/) to classify the language of a document (English, Spanish, Catalonian).

3. Create a confusion matrix to evaluate the performance of the classifier.
