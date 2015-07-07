class ItemItemRecommender(object):
    def __init__(self):
        """Initializes the parameters of the model.
        """
        pass

    def fit(self):
        """Implements the model and fits it to the data passed as an
        argument.

        Stores objects for describing model fit as class attributes.
        """
        pass
    
    def _set_neighborhoods(self):
        """Gets the items most similar to each other item.

        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of 
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.

        You will call this in your fit method.
        """
        pass

    def pred_one_user(self):
        """Accept user id as arg. Return the predictions for a single user.
        
        Optional argument to specify whether or not timing should be provided
        on this operation.
        """
        pass
        
    def pred_all_users(self):
        """Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).

        Optional argument to specify whether or not timing should be provided
        on this operation.
        """
        pass

    def top_n_recs(self):
        """Takes user_id argument and number argument.

        Returns that number of items with the highest predicted ratings,
        after removing items that user has already rated.
        """
        pass
