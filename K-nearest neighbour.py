import pandas as pd 
  
import numpy as np 

from scipy import stats as st

# Define the k-nearest neighbours class

class KNN_Classifier():
  
    # Function to define k the number of Nearest neighbours required for classification
    def __init__(self, k, X_train, Y_train):
        self.k = k

        #store trainging data
        self.X_train = X_train
        self.Y_train = Y_train

        # Define number of training samples and number of features
        self.num_samples, self.num_features = X_train.shape

    def predict_classification(self, X_test):

        # Define the test data
        self.X_test = X_test

        self.num_test_samples, self.num_test_features = X_test.shape

        #initialize the Y prediction array
        self.Y_predictions = np.zeros(self.m_test)

        # make predictions

        for i in range(self.m_test):

            # take each row of data from the test data
            x = self.X_test[i]

            # find the nearest neighbours of this test data point, first define the nearest neighbours array
            neighbours = np.zeros(self.k)

            neighbours = self.find_nearest_neighbours(x)
    
            # Now Classify
            self.Y_predictions[i] = st.mode(neighbours)[0][0]  


    # Function to find the nearest neighbours

    def find_nearest_neighbours(self, x):
        
        # define the distances array
        euclidean_distances = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):

            # Add distances to the array at index o
            euclidean_distances[i] = self.calculate_euclidean_distance(x, self.X_test[i])

        # sort the above algorithm in increasing order using .argsort(). This returns the indices of the sorted algorithm

        sorted_indices = euclidean_distances.argsort()

        Y_train_sorted = self.Y_train[sorted_indices]

        return Y_train_sorted[:self.k]
    
    # Function to calculate the Euclidean Distance between two input points

    def calculate_euclidean_distance(self, x, X_train):
        
        # Implementation of the Euclidean Distance Equation.
        return np.sqrt( np.sum( np.square(x - X_train) ) )



            


