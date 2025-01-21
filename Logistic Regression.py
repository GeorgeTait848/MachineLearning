import numpy as np
import pandas as np

class Logistic_Regression_Classifier():

    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        # get the number of features and input data

        self.num_of_samples, self.num_of_features = X_train.shape
    
    def sigmoid_function(self, Z):
        return 1 / 1 + np.exp(-Z)

    def gradient_of_log_likelyhood(self, theta, X, Y ):
        Z = np.dot(theta, X)
        return np.dot(X.T, Y - self.sigmoid_function(Z))

    def training_algoritm(self, learing_rate, num_iterations):
        
        self.theta = np.zeros(self.num_of_features)
        
        for i in range(num_iterations):
            self.theta += learing_rate * self.gradient_of_log_likelyhood(self.theta, self.X_train, self.Y_train)
        return self.theta
    
    def make_predictions(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test

        self.num_test_samples, self.num_test_features = np.shape(X_test)

        # Define the Predictions list
        self.Y_predictions = np.zeros(self.num_test_samples)

        # Calculate Z
        Z = np.dot(self.theta, self.X_test)

        # Make predictions
        self.Y_predictions = self.sigmoid_function(Z)

        return self.Y_predictions
    
    def loglikelyhood_function(self, X, Y):
        Z = np.dot(self.theta, X)
        return np.sum(Y * np.log(self.sigmoid_function(Z)) + (1 - Y) * np.log(1 - self.sigmoid_function(Z)) )