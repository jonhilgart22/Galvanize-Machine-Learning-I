from collections import Counter
import numpy as np

class TreeNode(object):
    '''
    A node class for a decision tree.
    '''
    def __init__(self):
        self.column = None  # (int)    index of feature to split on
        self.split_value = None  # value of the feature to split on
        self.categorical = True  # (bool) whether or not node is split on
                                 # categorial feature
        self.name = None    # (string) name of feature (or name of class in the
                            #          case of a list)
        self.left = None    # (TreeNode) left child
        self.right = None   # (TreeNode) right child
        self.leaf = False   # (bool)   true if node is a leaf, false otherwise
        self.classes = Counter()  # (Counter) only necessary for leaf node:
                                  #           key is class name and value is
                                  #           count of the count of data points
                                  #           that terminate at this leaf

    def predict(self, X):
        results = []
        for x in X:
            results.append(self.predict_one(x))
        return np.array(results)


    def predict_one(self, x):
        if self.leaf:
            return self.classes.most_common(1)[0][0]
        col_value = x[self.column]
        if self.categorical:
            if col_value == self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)
        else:
            if col_value < self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)


    def as_string(self, level=0, prefix=""):
        '''
        INPUT: TREENODE, INT, STRING
        OUTPUT: STRING

        Return a string representation of the tree rooted at this node.
        '''
        result = ""
        if prefix:
            indent = "  |   " * (level - 1) + "  |-> "
            result += indent + prefix + "\n"
        indent = "  |   " * level
        result += indent + "  " + str(self.name) + "\n"
        if not self.leaf:
            if self.categorical:
                left_key = str(self.value)
                right_key = "no " + str(self.value)
            else:
                left_key = "< " + str(self.value)
                right_key = ">= " + str(self.value)
            result += self.left.as_string(level + 1, left_key + ":")
            result += self.right.as_string(level + 1, right_key + ":")
        return result


    def __str__(self):
        return self.as_string()