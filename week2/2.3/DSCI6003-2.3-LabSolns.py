
# 2.3 Lab Solutions

# Part 1
# 1.1

import pandas as pd
df = pd.read_csv('data/grad.csv')

# 1.2

print df.describe()

# 1.3

import matplotlib.pyplot as plt
# %matplotlib inline

admit = pd.crosstab(df['admit'], df['rank'], rownames=['admit'])
(admit / admit.apply(sum)).plot(kind="bar")
plt.show()

# 1.4

df.hist()
plt.show()


# Part 2
# 2.1

from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant

X = df[['gre', 'gpa', 'rank']].values
X_const = add_constant(X, prepend=True)
y = df['admit'].values

logit_model = Logit(y, X_const).fit()

# 2.2

logit_model.summary()

# 2.3

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

kfold = KFold(len(y))

accuracies = []
precisions = []
recalls = []

for train_index, test_index in kfold:
    model = LogisticRegression()
    model.fit(X[train_index], y[train_index])
    y_predict = model.predict(X[test_index])
    y_true = y[test_index]
    accuracies.append(accuracy_score(y_true, y_predict))
    precisions.append(precision_score(y_true, y_predict))
    recalls.append(recall_score(y_true, y_predict))

print "accuracy:", np.average(accuracies)
print "precision:", np.average(precisions)
print "recall:", np.average(recalls)

# 2.4

dummies = pd.get_dummies(df['rank'], prefix='rank')
X2 = df[['gre','gpa']].join(dummies.ix[:,'rank_2':]).values

# 2.5

accuracies = []
precisions = []
recalls = []

for train_index, test_index in kfold:
    model = LogisticRegression()
    model.fit(X2[train_index], y[train_index])
    y_predict = model.predict(X2[test_index])
    y_true = y[test_index]
    accuracies.append(accuracy_score(y_true, y_predict))
    precisions.append(precision_score(y_true, y_predict))
    recalls.append(recall_score(y_true, y_predict))

print "accuracy:", np.average(accuracies)
print "precision:", np.average(precisions)
print "recall:", np.average(recalls)

# Part 3
# 3.1

from itertools import izip

model = LogisticRegression()
model.fit(X, y)

for name, coef in izip(df.columns[1:], model.coef_[0]):
    print "%s: %.4f" % (name, coef)

# 3.2

from math import exp

for i, coef in enumerate(model.coef_[0]):
    print "beta%d: %.5f" % (i + 1, exp(coef))

# 3.3

# For every 1 point increase in GRE score, the chance of getting in
# increases by a factor of 1.00174 on average.

# For every 1 point increase in GPA score, the chance of getting in 
# increases by a factor of 1.47499 on average.

# For every 1 point increase in school's rank (meaning 1 rank lower), 
# the chance of getting in decreases by a factor of 0.47904 on average.


# 3.4

from math import log

for i, coef in enumerate(model.coef_[0]):
    print "beta%d: %.5f" % (i + 1, log(2) / coef)


# Part 4
# 4.1

gre = df['gre'].mean()
gpa = df['gpa'].mean()
feature_matrix = []
ranks = [1, 2, 3, 4]
for rank in ranks:
	feature_matrix.append([gre, gpa, rank])

X_rank = np.array(feature_matrix)

# 4.2

probabilities_rank = model.predict_proba(X_rank)[:, 1]
for rank, prob in izip(ranks, probabilities_rank):
    print "rank: %d, probability: %f, odds: %f" % (rank, prob, prob / (1 - prob))


# 4.3

plt.plot(ranks, probabilities_rank)
plt.xlabel("rank")
plt.ylabel("probability")
plt.title("Affect of modifying the rank on probability of acceptance")
plt.show()

# 4.4

odds_rank = probabilities_rank / (1 - probabilities_rank)
plt.plot(ranks, odds_rank)
plt.xlabel("rank")
plt.ylabel("odds")
plt.title("Affect of modifying the rank on odds of acceptance")
plt.show()

# 4.5

plt.plot(ranks, np.log(odds_rank))
plt.xlabel("rank")
plt.ylabel("log probability")
plt.title("Affect of modifying the rank on log of odds of acceptance")
plt.show()

# 4.6

gpa = df['gpa'].mean()
rank = df['rank'].mean()
feature_matrix = []
gres = range(df['gre'].min(), df['gre'].max() + 1)
for gre in gres:
	feature_matrix.append([gre, gpa, rank])

X_gre = np.array(feature_matrix)

probabilities_gre = model.predict_proba(X_gre)[:, 1]
for gre, prob in izip(gres, probabilities_gre):
    print "gre: %d, probability: %f, odds: %f" % (gre, prob, prob / (1 - prob))

plt.plot(gres, probabilities_gre)
plt.xlabel("gre")
plt.ylabel("probability")
plt.title("Affect of modifying the GRE on probability of acceptance")
plt.show()

odds_gre = probabilities_gre / (1 - probabilities_gre)
plt.plot(gres, odds_gre)
plt.xlabel("gre")
plt.ylabel("odds")
plt.title("Affect of modifying the GRE on odds of acceptance")
plt.show()

plt.plot(gres, np.log(odds_gre))
plt.xlabel("gre")
plt.ylabel("log odds")
plt.title("Affect of modifying the GRE on log of odds of acceptance")
plt.show()


gre = df['gre'].mean()
rank = df['rank'].mean()
feature_matrix = []
gpas = range(int(np.floor(df['gpa'].min())), int(np.ceil(df['gpa'].max() + 1)))
for gpa in gpas:
	feature_matrix.append([gre, gpa, rank])

X_gpa = np.array(feature_matrix)

probabilities_gpa = model.predict_proba(X_gpa)[:, 1]
for gpa, prob in izip(gpas, probabilities_gpa):
    print "gpa: %d, probability: %f, odds: %f" % (gpa, prob, prob / (1 - prob))

plt.plot(gpas, probabilities_gpa)
plt.xlabel("GPA")
plt.ylabel("probability")
plt.title("Affect of modifying the GPA on probability of acceptance")
plt.show()

odds_gpa = probabilities_gpa / (1 - probabilities_gpa)
plt.plot(gpas, odds_gpa)
plt.xlabel("gpa")
plt.ylabel("odds")
plt.title("Affect of modifying the GPA on odds of acceptance")
plt.show()

plt.plot(gpas, np.log(odds_gpa))
plt.xlabel("gpa")
plt.ylabel("log odds")
plt.title("Affect of modifying the GPA on log of odds of acceptance")
plt.show()


