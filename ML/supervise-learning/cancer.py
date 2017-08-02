from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

cancer = load_breast_cancer()

#print("Cancer.keys():\n{}".format(cancer.keys()))
#print cancer['DESCR']
#cancer.keys():
#print cancer.data.shape
#print cancer.target
#print "Sample counts per class : \n"
#print { n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}
#print cancer.feature_names
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)
training_accuracy = []
test_accuracy = []
from sklearn.neighbors import KNeighborsClassifier
neighbors_settings = range(1,10)
print neighbors_settings
for n_neighbors in neighbors_settings:
    """ Build the Model"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    """ Record Training set accuracy """
    training_accuracy.append(clf.score(X_train,y_train))
    """ Record Generalization accuracy """
    test_accuracy.append(clf.score(X_test,y_test))
print training_accuracy
import matplotlib.pyplot as plt
plt.plot(neighbors_settings,training_accuracy,label = 'Training Accuracy')
plt.plot(neighbors_settings,test_accuracy,label= 'Testing Accuracy')
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()