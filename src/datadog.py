import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def plot(pandaframe):
    pandaframe.hist(bins=50, figsize=(20,15))
    plt.savefig("attribute_histogram_plots")
    plt.show()

def getBursts(pandaframe):
    rockbursts = []
    count = 0

    for index, bclass in enumerate(pandaframe['class']):
        if bclass == b'1':
            rockbursts.append(index)
    for index in rockbursts:
        print(pandaframe.iloc[index])
        count = count + 1

    print(count)

def prepareData(pandaframe):
    cat_to_num(pandaframe)
    xtrain, x_test, y_train, y_test = split_train_test(pandaframe)
    return normalizedata(xtrain, x_test, y_train, y_test)

def normalizeinput(pandaframe, normalizer):
    cat_to_num(pandaframe)
    skewed = ['genergy', 'gpuls', 'energy', 'maxenergy']
    # replaces 0s with an extremely small float
    epsilon = np.finfo(float).eps
    for col in skewed:
        pandaframe[col] = np.log(pandaframe[col].replace(0, epsilon))

    normalizedinput = normalizer.transform(pandaframe)

    return normalizedinput


def cat_to_num(df):
    #df['class'] = df['class'].map({b'1': 1, b'0': 0})

    hazard_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df['seismic'] = df['seismic'].map(hazard_map)
    df['seismoacoustic'] = df['seismoacoustic'].map(hazard_map)
    df['ghazard'] = df['ghazard'].map(hazard_map)
    df['shift'] = df['shift'].map({'W': 0, 'N': 1})


