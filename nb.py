import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

class NaiveBayes:

    def fit(self, X, y):
        rows, cols = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Mean, Variance and Prior
        self.mean = np.zeros((n_classes, cols), dtype = np.float64)
        self.var  = np.zeros((n_classes, cols), dtype = np.float64)
        self.priors = np.zeros(n_classes, dtype = np.float64)

        for i in self._classes:
            X_classes = X[i==y]
            self.mean[i,:] = X_classes.mean(axis=0)
            self.var[i,:] = X_classes.var(axis = 0)
            self.priors[i] = X_classes.shape[0] / float(rows)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []

        for i,j in enumerate(self._classes):
            prior = np.log(self.priors[i])
            class_conditional = np.sum(np.log(self.pdf(i, x)))
            pos = prior + class_conditional
            posteriors.append(pos)

        return self._classes[np.argmax(posteriors)]

    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numer = np.exp(-(x-mean)**2 / (2 * var))
        denom = np.sqrt(2* np.pi * var)
        return numer/ denom

def accuracy(y, y_pred):
    accuracy = np.sum(y == y_pred) / len(y)
    return accuracy

if __name__ =='__main__':
    X, y = datasets.make_classification(n_samples = 100000, n_features = 10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Accuracy: ", accuracy(y_test, predictions))


