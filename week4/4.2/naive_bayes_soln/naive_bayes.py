from __future__ import division
from collections import Counter, defaultdict
import numpy as np
import itertools

class NaiveBayes(object):
    def __init__(self, alpha=1):
        """
        INPUT:
        -alpha: float, laplace smoothing constant.
        """
        self.class_totals = defaultdict(int)
        self.class_feature_totals = defaultdict(Counter)
        self.class_counts = None
        self.alpha = alpha
        self.p = None

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        Compute the totals for each class and the totals for each feature
        and class.
        '''

        self.class_totals = defaultdict(int)
        self.class_feature_totals = defaultdict(Counter)
        for row, label in zip(X, y):
            word_counts = Counter(row)
            self.class_totals[label] += sum(word_counts.itervalues())
            self.class_feature_totals[label].update(word_counts) 

    def fit(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT: None
        '''
        self.class_counts = Counter(y)
        self.p = len(set(itertools.chain(*X)))
        self._compute_likelihood(X, y)

    def posteriors(self, X):
        '''
        INPUT:
        - X: List of list of tokens.

        OUTPUT: A list of counters. The keys of the counter
        will correspond to the possible labels, and the values
        will be the likelihood. (This is so we can use
        most_common method in predict).
        '''
        
        results= []
        likelihoods = {}
        p = self.p
        for row in X:
            row_counts = Counter(row)
            for label in self.class_totals:
                numerators = self.class_feature_totals[label]
                denominator = (self.class_totals[label] + self.alpha * p)

                #Start likelihood with log(P(y))
                likelihoods[label] = self.classs_counts[label]
                likelihoods[label] /= sum(self.class_counts.itervalues())
                likelihoods[label] = np.log(likelihoods[label])

                #Now add all the P(x_j|Y) terms.
                for value in row_counts:
                    likelihoods[label] += row_counts[value]*np.log(
                        (numerators[value] + self.alpha)/denominator
                    )
                    
            results.append(Counter(likelihoods))
        return results

    def predict(self, X):
        """
        INPUT:
        - X A list of lists of tokens.
        
        OUTPUT:
        -predictions: a numpy array with predicted labels.
        
        """
        return np.array([label.most_common(1)[0][0]
                         for label in self.posteriors(X)])

    def score(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent predicted correctly.
        '''

        return sum(self.predict(X) == y) / float(len(y))

