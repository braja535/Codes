from ML import mglearn
import matplotlib.pyplot as plt
X , y = mglearn.datasets.make_forge()

#plot dataset

#print X , y
mglearn.discrete_scatter(X[:,0],X[:,1],y)
#plt.legend(["Class 0","Class 1"],loc=8)
#plt.xlabel("First feature")
#plt.ylabel("Second Feature")
#print(X[:,0])
#print y
#plt.show()

#print ("X shape {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=1)
#plt.show()
"""  Split the data into training and test set  """
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

""" import and instanciate the class """

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


"""   Fit the classifier using traing set [storing the dataset in case of KNNclassifier ] """

clf.fit(X_train,y_train)

"""
call predict method -- for each datapoint in test set this computes the nearest neighbour in
training set and finds the most commo class among these  """

print ("Test set prediction:{}".format(clf.predict(X_test)))


""" test the score of our prediction"""

print ("Test set accurracy: {:.2f}".format(clf.score(X_test,y_test)))



fig, axes = plt.subplots(1,5, figsize=(10, 3))
for n_neighbors, ax in zip([1,3,4,6,9], axes):
# the fit method returns the object self, so we can instantiate
# and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=5)
plt.show()
