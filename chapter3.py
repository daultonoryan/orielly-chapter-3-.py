import numpy as np
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import mldata
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve


# importing the mlist dataset
MNISTdata = mldata.fetch_mldata("MNIST original")
X, y = MNISTdata["data"], MNISTdata["target"]
print("the shape of the features array is")
print(X.shape)
print("and the shape of the labels array is")
print(y.shape)

# making an optional function to see the MNIST images
def display_num_image():
    gen = random.Random()
    gen.seed(42)
    random_num = int(gen.random() * 70000)
    some_digit = X[random_num]
    some_digit_map = some_digit.reshape(28, 28)
    plt.imshow(some_digit_map, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


# setting up training and testing sets and randomizing the training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# bianary classifier testing with gradient decent
y_train5 = (y_train == 5)
y_test5 = (y_test == 5)

gradient_clf = SGDClassifier(random_state=42)
gradient_clf.fit(X_train, y_train5)
gradient_clf.predict(X[30000].reshape(1, -1))


# evaluation of model effectiveness using long form cross validation score
def get_long_form_CVscore():
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train5):
        cloneclf = clone(gradient_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train5[train_index]
        X_test_folds = X_train[test_index]
        y_test_folds = y_train5[test_index]
        cloneclf.fit(X_train_folds, y_train_folds)
        y_pred = cloneclf.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print("the accuracy of the initial model is %s" %str(n_correct/len(y_test_folds)))
        return y_pred


# prints the long and short way to do cross validaition accuracy score
y_pred = get_long_form_CVscore()
print("cross validation score on gradient classifier")
print(cross_val_score(gradient_clf, X_train, y_train5, cv=3, scoring="accuracy"))


# sets up a base estimator for comparison that predicts no fives and then takes cross validation score for that
# estimator and prints it
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never5 = Never5Classifier()
print("cross validation score on base estimator")
print(cross_val_score(never5, X_train, y_train5, cv=3, scoring="accuracy"))
y_train_pred = gradient_clf.predict(X_train)

# displays a confusion matrix
con = confusion_matrix(y_train5, y_train_pred)
print("confusion matrix")
print(con)

# using sklearn built in functions to calculate precision and recall, then for extra fun writing a function to do it
# from the confusion matrix
recall = recall_score(y_train5, y_train_pred)
precision = precision_score(y_train5, y_train_pred)
print("the precision score is %s and the recall score is %s \n" %(precision, recall))

FN = con[1, 0]
FP = con[0, 1]
TP = con[1, 1]
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print("the precision score is %s and the recall score is %s" %(precision, recall))

# implementing sklearn f1 score the f1 score is used to combine precision and recall
print("the f1 score is %s" %f1_score(y_train5, y_train_pred))

# changing threshold values to show the affect on precision and recall
y_scores = gradient_clf.decision_function(X[30000].reshape(1, -1))
print(y_scores)


# simple function to get a thresholded array takes a y_
def thresholded_array(threshold, y_scores = gradient_clf.decision_function(X[30000].reshape(1, -1))):
    y_some_digit_pred = (y_scores > threshold)
    print(y_some_digit_pred)


print("demonstrating the precision recal tradeoff by altering prediction arrays using thresholding")
print(thresholded_array(0), thresholded_array(200000))
# demonstrating how to choose an effective threshold
y_scores = cross_val_predict(gradient_clf, X_train, y_train5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train5, y_scores)

print(precisions, recalls, thresholds)
def precision_recall_graph(precisionss, recallss, thresholdss):
    plt.plot(thresholdss, precisionss[:-1], "b--", label="precision")
    plt.plot(thresholdss, recallss[:-1], "g-", label="recall")
    plt.xlabel("threshold")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)


precision_recall_graph(precisions, recalls, thresholds)
plt.show()





# the following lines display graphical representations the number images function starts on line 15,
display_num_image()





