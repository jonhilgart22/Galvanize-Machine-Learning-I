# Model Comparison

You've learned a lot of techniques. Today is about comparing and contrasting
those techniques; your work today will serve as a foundation for future case studies and potentially your project.

Here are two lists of business use cases for data science. Skim through them to
motivate your learning today. 

* [Kaggle Data Science Use Cases](http://www.kaggle.com/wiki/DataScienceUseCases)
* [Industry use case list](http://thebigdatainstitute.wordpress.com/2013/05/23/our-favorite-40-big-data-use-cases-whats-your/)

There are a list of resources at the bottom that you should feel free to consult
during your research.

## Your Task

Together with your group, you will design an informal "decision tree" that will
map a use case to a technique (or some combination of techniques). You will
think about and generate the relevant questions to ask about the real world use
case in order to ultimately decide on how you will model the problem.

This exercise is a rapid research project. There should be much debate and discussion
today. Search, read, think, talk, repeat.

First, you'll investigate how different models handle various dataset
attributes, constraints, etc. Then you'll construct a decision tree to help you
select which model to use given a new problem.

### Phase I: Comparisons

Answer the questions for each model/algorithm.

As you go through these questions, focus on "Why?" (i.e. not just "Can I do
online learning with this model?", but also "Why can I do online learning with
this model?"). Understand the guts of things.

1. Classification algorithms
  * What are your options?
  * What is the Hypothesis, Cost function, and Optimization technique?
  * How do the various approaches compare in:
    * Dealing with extremely high-dimensional space (lots of features)
    * Dealing with large amounts of data (lots of examples). How about datasets
      that are so large they cannot fit into memory?
    * Training performance: how long does it take to train?
    * Prediction time: how long does it take to predict?
    * Interpretability: does the model provide insights about what matters,
      correlations, or causes?
    * Communication: how would you communicate the model results? How would you
      describe briefly to an interested, smart, but non-technical stakeholder
      what the model is doing (e.g. your product manager or CEO)?
    * Visualization: how could you visualize the model or its results/findings?
    * Evaluation: what are your options for model evaluation? Why would you use
      one evaluation metric over another (pros and cons of the various
      evaluation metrics/methods)?
    * Nonlinearity: can the model (out of the box or modified) find nonlinear
      decision boundaries? What are the tradeoffs, if any, of this additional
      power (i.e. why not always use nonlinear models)?
    * Fewer examples than features (`n << p`)
    * Sensitivity to outliers: how sensitive is the model to outliers and how
      would you deal with outliers?
    * Overfitting: how liable is it to overfit? How can you tell if the model is
      overfitting? What are your options to reduce overfitting? 
    * Hyperparameters: how many knobs are there to turn, what are they, and what
      do they do?
    * Generative vs. discriminative: does the model model the causal/generative
      process or does it just differentiate? Can the model produce example data?
      What are the advantages of a generative model? Disadvantages?
    * Online learning: can the model be used in an online setting (i.e. can it
      train on new examples as they come in or does it have to be trained on all
      the training data at once)?
    * Unique attributes or gotchas: what are some important features of this
      model in theory or practice? Some examples: SVM is a large-margin
      classifier, or make sure you scale your features for kNN or kMeans. Any
      tricks or gotchas that come up in implementation?
    * Special use cases: are there domains or use cases to which this model is
      particularly well-suited (e.g. Naive Bayes for text classification)?
    * Unbalanced classes: how does this model deal with unbalanced classes? What
      can you do to improve performance in the face of unbalanced classes?

1. Regression algorithms
  * Same questions as above

1. Unsupervised learning
  * Interpretability: how would you interpret the findings of an unsuperivsed
    learning algorithm? How could you interpret clusters or latent variables?
  * Visualization: how would you visualize the findings of an unsupervised
    learning algorithm?
  * Integration with supervised techniques: how can the method be used in a
    supervised learning context? How can it be used in feature engineering? When
    or why would you use it in a supervised context?

1. Natural language processing
  * Featurization: what are the various ways of featurizing text? What are the
    differences (i.e. why featurize in one way or another?)?

### Phase II: Decision Tree

Now that you've investigated the various techniques, construct an informal
decision tree that you could use on a real world data problem to decide what
techniques to use.

Use your findings from Phase I to motivate the "splits." It would make sense to
start with "Supervised / Unsupervised" as the first split, followed by
"Classification / Regression" under "Supervised" as a second split, but if you
come up with something more interesting, feel free to go with it.

## Techniques

These are a list of the techniques we've covered in class.

* Supervised learning
  * Classification
    * Logistic regression
    * Support vector machine
    * Decision trees
    * Random forest
    * kNN
    * Naive Bayes
    * AdaBoost
    * Bagging
  * Regression
    * Linear regression
    * Polynomial regression
    * Decision tree regression

* Unsupervised learning
  * PCA/SVD
  * NMF
  * k-means/PAM clustering
  * GMM

* Natural Language
  * Bag of words
  * tf-idf
  * etc.
  * etc.

* Other
  * Regularization
  * Scaling/Feature Engineering
  * Cross-validation
    * F_1
    * Precision
    * Recall
    * ROC
  * Clustering Evaluation
    * Elbow Method
    * Silhouette
    * Gap Statistic

## Resources

* [Google](http://www.google.com)
* [Machine Learning Overview](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
* [Data Scientist - the job](http://datamyze.com/blog/wtf-does-a-data-scientist-do-all-day/)
* [sklearn Cheatsheet](https://raw.githubusercontent.com/amueller/sklearn_tutorial/master/cheat_sheet.png)
* [Quora - classifiers](http://www.quora.com/What-are-the-advantages-of-different-classification-algorithms)
* Books (on Time Capsule)
  * Doing Data Science: Chapter 13, Lessons from Data Competitions: Data Leakage
    and Model Evaluation
  * Programming Collective Intelligence: Chapter 12: Algorithm Summary, Appendix
    B: Mathematical Formulas
  * Machine Learning For Hackers: Chapter 12: Model Comparison
  * Machine Learning in Action: Chapter 1: Classification
  * Data Science for Business
