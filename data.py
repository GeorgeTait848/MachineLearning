import pandas as pd
from pandas import DataFrame
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def fetch_MEGA_dataset() -> DataFrame: 
    col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAssym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "isGamma"]
    dat = pd.read_csv('MAGIC/magic04.data', names=col_names)

    #classifying gamma as 1, hadron as 0. 
    dat['isGamma'] = (dat["isGamma"] == 'g').astype(int)

    return dat

def split_dataset(df: DataFrame, train_frac = 0.6, valid_frac = 0.2, test_frac = 0.2) -> tuple[DataFrame, DataFrame, DataFrame]: 
    '''Splits a dataset into sections for training, validation and testing. Default values are 60% for training, 20% for validation, 
    20% for testing.
    
    Returns the training, validation and test data'''

    assert(train_frac + valid_frac + test_frac == 1.0)

    df_len = len(df)
    train_split_point = int(train_frac*df_len)
    valid_split_point = int((train_frac + valid_frac)*df_len)

    training_data, validation_data, test_data = np.split(df.sample(frac=1), [train_split_point, valid_split_point])

    return DataFrame(training_data, columns=df.columns.values), DataFrame(validation_data, columns=df.columns.values), DataFrame(test_data, columns=df.columns.values)


def oversample_data(features: np.ndarray, outcomes: np.ndarray):

    random_oversampler = RandomOverSampler()
    oversampled_features, oversampled_outcomes = random_oversampler.fit_resample(features, outcomes)

    return oversampled_features, oversampled_outcomes

def scale_dataset(df: DataFrame, oversample = False) -> DataFrame:
    '''Scales values around mean value for each feature using StandardScaler from scikit-learn'''

    features = df[df.columns[:-1]].values
    outcomes = df[df.columns[-1]].values

    scaler = StandardScaler()

    features = scaler.fit_transform(features)

    if oversample: 
        oversampled_features, oversampled_outcomes = oversample_data(features, outcomes)
        oversampled_dat = np.hstack((oversampled_features, np.reshape(oversampled_outcomes, (-1, 1))))
        return DataFrame(oversampled_dat, columns=df.columns.values)


    dat = np.hstack((features, np.reshape(outcomes, (-1,1))))
    return DataFrame(dat, columns=df.columns.values)

def get_prepared_data():

    df = fetch_MEGA_dataset()
    train, valid, test = split_dataset(df)

    training_data = scale_dataset(train, oversample=True)
    validation_data = scale_dataset(valid)
    testing_data = scale_dataset(test)

    return training_data, validation_data, testing_data


def separate_features_outcomes(df: pd.DataFrame):

    features = df[df.columns[:-1]].values
    outcomes = df[df.columns[-1]].values

    return features, outcomes

def plot_MEGA_data_normalised_histogram(): 
    '''Plotting the different features as normalised histograms for Gamma particles and Hadrons'''

    dat = fetch_MEGA_dataset()

    for feature in dat.columns.values[:-1]:

        plt.hist(dat[dat["isGamma"] == 1][feature], color='blue', label="gamma", alpha=0.7, density=True)
        plt.hist(dat[dat["isGamma"] == 0][feature], color='red', label="hadron", alpha=0.7, density=True)
        plt.title(feature)
        plt.xlabel(feature)
        plt.ylabel("Probability")
        plt.legend()
        plt.show()


def main():
    train, valid, test = get_prepared_data()
    print(train)
    print(len(train[train["isGamma"]==1]))
    print(len(train[train["isGamma"]==0]))
    



if __name__ == "__main__":
    main()

