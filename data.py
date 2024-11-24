import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAssym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "isGamma"]
dat = pd.read_csv('MAGIC/magic04.data', names=col_names)

#classifying gamma as 1, hadron as 0. 
dat['isGamma'] = (dat["isGamma"] == 'g').astype(int)

print(dat)


#plottting the different features as normalised histograms for Gamma particles and Hadrons

for feature in col_names[:-1]:

    plt.hist(dat[dat["isGamma"] == 1][feature], color='blue', label="gamma", alpha=0.7, density=True)
    plt.hist(dat[dat["isGamma"] == 0][feature], color='red', label="hadron", alpha=0.7, density=True)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel("Probability")
    plt.legend()
    plt.show()