import pandas as pd
from sklearn.datasets import load_iris

"""  load iris_data to dataset """
iris_dataset = load_iris()
#print type(iris_dataset)
#print iris_dataset.items()
#print("Keys of iris_dataset\n{}".format(iris_dataset.items()))
#print (iris_dataset['DESCR'])
""" featurenames in loaded dataset """
#print ("fature_names : {}".format(iris_dataset['feature_names']))

""" the target names """
#print ("target_name : {}".format(iris_dataset['target_names']))

''' np arrays shape or dimensions '''
#print ("Np array [data] dimensions {}".format(iris_dataset['data'].shape))

""" keys in dataset  """
#print (iris_dataset.keys())

"""  no.of elements in iris_dataset  """
#print ("First five columns of data:\n {}".format(iris_dataset['data'][:5]))

#print ("First five columns of data:\n {}".format(iris_dataset['target'].shape))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0
)
#print y_test
#print y_train
from ML import mglearn
import numpy as np
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                                 hist_kwds ={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
import matplotlib.pyplot as plt
plt.show()
#print grr

#plt.plot(grr[0])

#%matplotlib inline
#import matplotlib.pyplot as plt
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
#x = np.linspace(-10, 10, 100)
# Create a second array using sine
#y = np.sin(x)
# The plot function makes a line chart of one array against another
#plt.plot(x, y, marker="x")
#plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
                                   iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print ("{}".format(y_pred))
print (np.mean(y_pred == y_test))

print (knn.score(X_test,y_test))