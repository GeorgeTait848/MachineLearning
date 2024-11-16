import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAssym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "isGamma"]
dat = pd.read_csv('MAGIC/magic04.data', names=col_names)

#classifying gamma as 1, hadron as 0. 
dat['isGamma'] = (dat["isGamma"] == 'g').astype(int)

print(dat)