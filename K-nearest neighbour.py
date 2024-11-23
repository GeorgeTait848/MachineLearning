import pandas as pd 
  
import numpy as np 

# Define the class

class K_Nearest_Neighbours_Classifier():
  
    # Function to define K the number of Nearest neighbours required for classification
    def __init__(self, K):

        self.K = K
    
    # Function to store the data set ("train"... not really)

    def Training_Data(self, X_train, Y_train):
        
        self.X_train = X_train
        self.Y_train = Y_train

        # Define number of training samples (m) and number of features (n)
        self.m, self.n = X_train.shape
    
    # Function to predict the classification

    def prediction(self, X_test):

        # Define the test data
        self.X_test = X_test

        # Define number of testing samples and number (m_test) and number of testing features (n_test)
        self.m_test, self.n_test = X_test.shape

        #initialize the Y prediction array
        self.Y_predict = np.zeros(self.m_test)

        # make predictions

        for i in range(self.m_test):

            # take each row of data from the test data
            x = self.X_test[i]

            # find the nearest neighbours of this test data point, first define the nearest neighbours array
            Neighbours = np.zeros(self.K)

            Neighbours = self.Find_Nearest_Neighbours(x)

    # Function to find the nearest neighbours

    def Find_Nearest_Neighbours(self, x):
        
        # define the distances array
        Euclidean_Distances = np.zeros(self.m)
        
        for i in range(self.m):

            # Add distances to the array at index o
            Euclidean_Distances[i] = self.Calculate_Euclidean_Distance(x, self.X_test[i])

        # sort the above algorithm in increasing order using .argsort(). This returns the indices of the sorted algorithm

        Sorted_indices = Euclidean_Distances.argsort()

        Y_train_sorted = self.Y_train[Sorted_indices]

        return Y_train_sorted[:self.K]
    
    # Function to calculate the Euclidean Distance between two input points

    def Calculate_Euclidean_Distance(self, x, X_train):
        
        # Implementation of the Euclidean Distance Equation.
        return np.sqrt( np.sum( np.square(x - X_train) ) )



            


