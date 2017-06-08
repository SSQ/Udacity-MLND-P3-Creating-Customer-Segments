# Unsupervised Learning
## Project: Creating Customer Segments
### Files Description
- `project description.md`: Project overview, highlights, evaluation and software requirement. **Read Firstly**
- `README.md`: this file.
- `customer_segments.ipynb`: This is the main file where I will be performing your work on the project.
- `customers.csv`: The project dataset. I'll load this data in the notebook.
- `visuals.py`: A Python file containing visualization code that is run behind-the-scenes. **Not my work**
- `customer_segments.html`: `html` version of the main file.

### Run
#### 1.Want Modify
In a command window (OS: Win7), navigate to the top-level project directory that contains this README and run one of the following commands:
`jupyter notebook customer_segments.ipynb`
This will open the Jupyter Notebook software and project file in your browser.
#### 2.Just Have a Look
Double click `customer_segments.html` file. You can see this file in your browser.

## Project Implementation
### Data Exploration
#### Implementation: Selecting Samples
Three separate samples of the data are chosen and their establishment representations are proposed based on the statistical description of the dataset.
```
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [202,329,307]
```
#### Implementation: Feature Relevance
A prediction score for the removed feature is accurately reported. Justification is made for whether the removed feature is relevant.
- Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
- Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
    - Use the removed feature as your target label. Set a `test_size` of 0.25 and set a `random_state`.
- Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
- Report the prediction score of the testing set using the regressor's `score` function.
```
score = []
list_name = np.array(data.columns)
#print list_name
for label in list_name:
    # TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.copy()

    new_data.drop(label,axis = 1, inplace = True)
    #print new_data

    # TODO: Split the data into training and testing sets using the given feature as the target
    target = data[label]
    #print target 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_data, target,test_size=0.25, random_state=0)

    # TODO: Create a decision tree regressor and fit it to the training set
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)

    # TODO: Report the score of the prediction using the testing set
    print str(label) + ": " + str(regressor.score(X_test,y_test))
    score.append(regressor.score(X_test,y_test))
#print score
```
#### Feature Distributions
- Identifies features that are correlated and compares these features to the predicted feature. 
- Further discuss the data distribution for those features.

### Data Preprocessing
#### Implementation: Feature Scaling
Feature scaling for both the data and the sample data has been properly implemented in code.
- Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
- Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.
```
# TODO: Scale the data using the natural logarithm
log_data = np.log(data.copy())
#display(log_data)
# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples.copy())
```
#### Implementation: Outlier Detection
Identify extreme outliers and discusses whether the outliers should be removed. 

Justification is made for any data points removed.
- Assign the value of the 25th percentile for the given feature to `Q1.` Use `np.percentile` for this.
- Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
- Assign the calculation of an outlier step for the given feature to `step`.
- Optionally remove data points from the dataset by adding indices to the `outliers` list.
```
# For each feature find the data points with extreme high or low values
indix_outlier = []
for feature in log_data.keys():
    #print feature
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    #print Q1
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    indix_outlier.extend(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.tolist())
    #print indix_outlier
    
# OPTIONAL: Select the indices for data points you wish to remove
repeated_outliers=[]
for el in indix_outlier:
    if indix_outlier.count(el)>1:
        #print el, indix_outlier.count(el)
        repeated_outliers.append(el)
print "Following records are outliers for more than one feature:", list(set(repeated_outliers))

outliers  = list(set(indix_outlier))
print "Following records are outliers:", outliers
```
### Feature Transformation
#### Implementation: PCA
The total variance explained for two and four dimensions of the data from PCA is accurately reported. 

The first four dimensions are interpreted as a representation of customer spending with justification.
- Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
- Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`
```
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)
```
#### Implementation: Dimensionality Reduction
PCA has been properly implemented and applied to both the scaled data and scaled sample data for the two-dimensional case in code.
- Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
- Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
- Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.
```
# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)
```
### Clustering
#### Clustering Algorithm
The Gaussian Mixture Model and K-Means algorithms have been compared in detail. 

Choice of algorithm is justified based on the characteristics of the algorithm and data.
```
pros in K-Means
    simple, easy to implement;
    easy to interpret the clustering results;
    fast and efficient in terms of computational cost, typically O(Knd);
pros in GMM
    GMM model accommodates mixed membership
I choose k-means because of simple. In kmeans, a point belongs to one and only one cluster, whereas in GMM a point belongs to each cluster to a different degree.
```
#### Implementation: Creating Clusters
Several silhouette scores are accurately reported, and the optimal number of clusters is chosen based on the best reported score. The cluster visualization provided produces the optimal number of clusters based on the clustering algorithm chosen.
- Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
- Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
- Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
- Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
- Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
    - Assign the silhouette score to `score` and print the result.
```
# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.cluster import KMeans
for i in range(2,7):
    clusterer = KMeans(n_clusters=i, random_state=1)
    clusterer.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.cluster_centers_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    labels = clusterer.labels_
    from sklearn.metrics import silhouette_score
    score = silhouette_score(reduced_data, labels, metric='euclidean')
    print str(i) +': ' +str(score)   

#
clusterer = KMeans(n_clusters=2, random_state=1)
clusterer.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
labels = clusterer.labels_
score = silhouette_score(reduced_data, labels, metric='euclidean')
print score
```
#### Implementation: Data Recovery
The establishments represented by each customer segment are proposed based on the statistical description of the dataset. The inverse transformation and inverse scaling has been properly implemented and applied to the cluster centers in code.
- Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
- Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
```
# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)
```
#### Sample Predictions
Sample points are correctly identified by customer segment, and the predicted cluster for each sample point is discussed.
### Conclusion
#### A/B Test
Correctly identify how an A/B test can be performed on customers after a change in the wholesale distributor’s service.

Companies will often run A/B tests when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?
```
- From the introduction of A/B tests provided, we can modify the men and women to cluster0 and cluster1, replace Variant A and B to 5 days a week and 3 days a week. And the Total sends can change a little according to our data set. Collect the data in Total response and analysis it.
- That is to say we run A/B tests in each cluster, we have 2 clusters so we run A/B tests twice. In each cluster we need to choose two groups, control group and experiment group,consisting of many pairs of the most similar data points. Similar means the least distance.
```
#### Predicting Additional Data
Discuss with justification how the clustering data can be used in a supervised learner for new predictions.

Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a customer segment it best identifies with (depending on the clustering algorithm applied), we can consider 'customer segment' as an engineered feature for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a customer segment to determine the most appropriate delivery service.
How can the wholesale distributor label the new customers using only their estimated product spending and the customer segment data?
```
- From last question, we know how to choose delivery service and get the data set.
- use customer segment in a form of one-hot code.
- split data set into training and testing data set. Input is features and customer segment, output is the type of delivery service, 0 or 1.
- initial a supervised learner, fit and test it
- choose a good parameter
- predict ten new customers and get the type of delivery service.
```
#### Comparing Customer Data
Comparison is made between customer segments and customer ‘Channel’ data. Discussion of customer segments being identified by ‘Channel’ data is provided, including whether this representation is consistent with previous results.


