## Kmeans Applied
We have learned the intricacies of the kmeans algorithm, but we have not yet seen how it can be used.  For this exercise we will apply our kmeans algorithm to our NYT articles to discover topics.  Kmeans can be used for a number of different applications, but document clustering is a fairly frequent technique especially when you have no other labels/categories to go off of.  While we have section labels, we do not have topic labels.  The basic flow:

* Featurize your data however you so choose (text -> bag-o-words/tf-idf :: images -> bag-o-images, greyscaling, etc. :: user profiles -> vector of habits, demographics, cohort characteristics :: etc.)
* Apply kmeans clustering
    * A few times to find appropriate centroids and optimal value of k
* With best centroids, inspect their features
    * Centroids represent "average" data point of all points belonging to its cluster
* Centroid features are descriptive of the group as a whole
* Data points are representative as well, inspect a few of the data points for each cluster
    * Look at common properties among them

A great example to get a better sense of what the centroids represent is k-means [applied to images](http://nbviewer.ipython.org/github/temporaer/tutorial_ml_gkbionics/blob/master/2%20-%20KMeans.ipynb#$k$-Means-on-Images).  Applied to the MNIST image dataset, the centroids represent the 'average' digit:

![images](images/images.png)

The repo contains a 'articles.pkl' file that has 1405 articles from 'Arts','Books','Business Day', 'Magazine', 'Opinion', 'Real Estate', 'Sports', 'Travel', 'U.S.', and 'World'. This is a [pickled](https://docs.python.org/2/library/pickle.html) data frame and can be loaded back into a [data frame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_pickle.html#pandas.read_pickle).  You probably want to eventually get it out of pandas DataFrames when you perform your analysis.

| Section | count|
| :---| :--|
|Arts| 91|
|Automobiles| 5|
|Books| 37|
|Booming| 7|
|Business Day| 100|
|Corrections| 10|
|Crosswords & Games| 2|
|Dining & Wine| 19|
|Education| 4|
|Fashion & Style| 46|
|Great Homes and Destinations| 5|
|Health| 10|
|Home & Garden| 10|
|Magazine| 11|
|Movies| 28|
|N.Y. / Region| 92|
|Opinion| 84|
|Paid Death Notices| 11|
|Real Estate| 13|
|Science| 18|
|Sports| 134|
|Technology| 13|
|Theater| 16|
|Travel| 9|
|U.S.| 88|
|World | 131|
|Your Money | 6 |

1. Apply kmeans clustering to the `articles.pkl`. Use pandas' `pd.read_pickle()`.  Use either the kmeans you implemented or [scikit-learn's](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) module if you did not finish your implementation.

2. To find out what "topics" Kmeans has discovered we must inspect the centroids.  Print out the centroids of the Kmeans clustering.
   
   These centroids are simply a bunch of vectors.  To make any sense of them we need to map these vectors back into our 'word space'.  Think of each feature/dimension of the centroid vector as representing the "average" article or the average occurances of words for that cluster.

3. But for topics we are only really interested in the most present words, i.e. features/dimensions with the greatest representation in the centroid.  Print out the top ten words for each centroid.
  * Sort each centroid vector to find the top 10 features
  * Go back to your vectorizer object to find out what words each of these features corresponds to.

4. Look at the docs for `TfidfVectorizer` and see if you can limit the number of features (words) included in the feature matrix.  This can help reduce some noise and make the centroids slightly more sensible.  Limit the `max_features` and see if the words of the topics change at all.

5. An alternative to finding out what each cluster represents is to look at the articles that are assigned to it.  Print out the titles of a random sample of the articles assigned to each cluster to get a sense of the topic.

6. What 'topics' has kmeans discovered? Can you try to assign a name to each?  Do the topics change as you change k (just try this for a few different values of k)?

7. If you set k == to the number of NYT sections in the dataset, does it return topics that map to a section?  Why or why not?

8. Try your clustering only with a subset of the original sections.  Do the topics change or get more specific if you only use 3 sections (i.e. Sports, Art, and Business)?  Are there any cross section topics (i.e. a Sports article that talks about the economics of a baseball team) you can find? 

### Cluster Evaluation

I have uploaded the [Telco dataset](https://github.com/zipfian/DSCI6003-student/blob/master/week4/exercise/data/churn.csv) and a dataset of [congressional voting](https://github.com/zipfian/DSCI6003-student/blob/master/week4/exercise/data/congressional_voting.csv). Try to discover user segments/demographics from the dataset. As seen in lecture there are a few techniques to evaluate our clusters, we are lucky since there are also labels on these two datasets.  We will perform **external** as well as **internal** evaluation.

1. Apply both kmeans and `scikit-learn`s [GMM](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html) clustering to the datasets.  
2. For the Telco dataset use the `churn` column as the label.  For the congressional voting dataset, use the party affiliation as the label.  Varying `k` for both kmeans and GMM, plot the `purity` of the clusters as well as the **within cluster dispersion (WCD)**
3. From the **WCD** plot, use the elbow method to determine the optimal `k`
4. According the the purity, which `k` would be selected? Is this the same as the value found from the elbow method?
5. Is it correct to use the `churn` column and party affiliation in our `purity` calculation?  Why might this not be the best given what we are trying to achieve with our clustering?
2. Do any meaningful clusters come out from either? What are the characteristics of the user groups from kmeans (i.e. the 'average' user from each group)?
3. Use the [silhouette score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) to find the optimal `k`.  Is this the same or different than our above methods?
5. Run kmeans with this value.  Do the clusters come out to be similar to GMM with the same number of clusters clusters?

## Extra Credit

### Hierarchical Clustering
![dendrogram](images/sortingDendrogram.png)

We have been introduced to distance metrics and the idea of similarity, but we will take a deeper dive here. For many machine learning algorithms, the idea of 'distance' between two points is a crucial abstraction to perform analysis. For Kmeans we are usually limited to use Euclidean distance even though our domain might have a more approprite distance function (i.e. Cosine similarity for text).  With Hierarchical clustering we will not be limited in this way.   
We already have our bags and played around with Kmeans clustering.  Now we are going to leverage [Scipy](http://www.scipy.org/) to perform [hierarchical clustering](http://en.wikipedia.org/wiki/Hierarchical_clustering).

1. Hierarchical clustering is more computationally intensive than Kmeans.  Also it is hard to visualize the results of a hierarchical clustering if you have too much data (since it represents its clusters as a tree). Create a subset of the original articles by filtering the data set to contain at least one article from each section and at most around 200 total articles.

    One issue with text (especially when visualzing/clustering) is high dimensionality.  Any method that uses distance metrics is susceptible to the [curse of dimensionality](http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/). `scikit-learn` has some utility to do some feature selection for us on our bags.  

2. The first step to using `scipy's` Hierarchical clustering is to first find out how similar our vectors are to one another.  To do this we use the `pdist` [function](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) to compute a similarity matrix of our data (pairwise distances).  First we will just use Euclidean distance.  Examine the shape of what is returned.

3. A quirk of `pdist` is that it returns one looong vector.  Use scipy's [squareform](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html) function to get our long vector of distances back into a square matrix.  Look at the shape of this new matrix.

4. Now that we have a square similarity matrix we can start to cluster!  Pass this matrix into scipy's `linkage` [function](http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) to compute our hierarchical clusters.

5. We in theory have all the information about our clusters but it is basically impossible to interpret in a sensible manner.  Thankfully scipy also has a function to visualize this madness.  Using scipy's `dendrogram` [function](http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html) plot the linkages as a hierachical tree.

_Note: [Here](http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week3/hierarchical_clustering.ipynb) is a very simple example of putting all of the pieces together_


### Hierachical Topics
Now that we have our dendrogram we can begin exploring the clusters it has made.

1. To make your clusters more interpretable, change the labels on the data to be the titles of the articles. Can you find any interesting clusters or discover any topics not present in the NYT sections?  Are there any overlaps with the Kmeans topics and the hierarchical topics?

2. In addition, we might also be interested in how these hierachical clusters compare to the NYT sections.  Label each point not only with the title but also the NYT section it belongs to.  Do any cross section topics emerge?

    __Protip: You can output a hi-res [image](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig) with matplotlib to then view outside of IPython (which you can zoom in on).__

    ![articles.png](images/article_cluster.png)

3. Explore different clusterings on a per section basis. Perform the same analysis on each of the Arts, Books, and Movies sections (i.e. cluster one section at a time).

4. Repeat this process using cosine similarity (and if you want Pearson correlation and the Jaccard distance).  Read about scipys distance metrics [here](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist).  Why might cosine distance be better for clustering the words of our articles?

5. Compare the clusters returned with cosine and Euclidean distance metrics.

6. We have visualized similarity between articles, but we can also see which words are similar and co-occur.  This dendrogram is somewhat less-sensical, but lets look at it anyway.  First limit the number of features with the vectorizer (if you haven't already).  500-1000 words is probably the limit of what you can visualize effectively.  Transpose your feature matrix so now rows correspond to words and the columns correspond to the articles.

7. Perform the same analysis as above and inspect the dendrogram with the words from the articles.  Anything you wouldn't expect?

    ![words.png](images/words_cluster.png)



