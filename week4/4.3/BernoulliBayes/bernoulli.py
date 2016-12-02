# Put import statements here


class BernoulliBayes(object):

    def __init__(self):
        self.prior = {}
        self.pvc = {}

    def fit(self,X,y):
        '''
        Input: 
            - X: (2d numpy array) Contains input data
            - y: (numpy array) Contains labels
        Ouput: 
            - None, fit model to input data
        '''
        # for each class c in C
            # count all documents in D belonging to that class, Nc
            # update the prior[c] with Nc
            # for each word v in V
                # count all docs in D containing v belonging to that class, Ncv
                # add in the count to the conditional probability table P(v, c) = (N_{cv} + 1)/(Nc + 2)
        # store P(v,c) and Priors



    def predict(self,X):
        # For each point in X
            # for each class c in C
                    # initialize score[c] = log(prior[c])
                    # for all v in V:
                        # if v is 1:
                            # score[c] += log(P(v, c))
                        # else:
                            # score[c] += log(1 - P(v, c)
            # predict argmax(score[c])
        # Return predictions
         