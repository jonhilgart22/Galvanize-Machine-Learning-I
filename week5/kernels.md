## sprint additions

* linear logistic -> do not use mesh
* polynomial and RBF -> do use mesh, cannot other wise

## lecture additions:

* project feature space to higher dimensional space
* logistic regression is just linear regression with a logit kernel (GLM)

# Kernels

So far everything we have we have encountered can be turned into a fixed size feature vector: users, cars, etc.

What are we to do if we have less structured data: text, images, molecular structure?

One approach is to blow apart your feature space and encode every possible value by a new column: i.e. for text each possible word in your corpus (can be tens of thousands) will be a column.  What problems will we encounter?

* Sparse encodings (bag of words)
* Order often gets destroyed
* Computational can be intractable

Another answer to this problem is kernels!

## Like Popcorn?

In the abstract, think of a kernel as a function that maps a feature space: `f(x) -> x'`

But more than this, kernels represent the similarity between *two* feature vectors (assume `k(x1, x2) gte 0`), and creates a new transformed feature space from these similarities: `k(x1, x2) -> x'`

### Formalism

Let a kernel function be some real-valued function $kappa$ of two arguments ($kappa(x,x')$).  Also the function is typically symmetric: $kappa(x,x') = kappa(x',x)$ and non-negative ($kappa(x,x') gte 0$)

Because of these properties it can be interpreted as the similarity between its arguments but can also be more abstract as well.

#### Mercer Kernels

#### Feature Expansion

### Give me the Concrete

#### Linear

#### Polynomial

#### RBF

#### Other: String Kernel

Hopefully you are starting to realize the power (and generalizability) of kernels.  Because of this you can make (most anything) a kernel, and your decision is often dictated by the constraints of your problem.

One unique application of kernels is to the problem of text.

### Aside: Decision Boundaries

## Application

### Classification

#### kNN

#### Logistic Regression

#### What about new points?

### Regression

Kernels are a natural fit for distance based methods (kNN and clustering) or any model that uses the concept of similarity.  It is not too much of a stretch to apply them to parametric models (SVM) but before we get to SVMs let us try a simpler method.

#### Kernelized Ridge
