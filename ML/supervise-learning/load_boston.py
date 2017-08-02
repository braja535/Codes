from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from ML import mglearn
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))

X ,y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

