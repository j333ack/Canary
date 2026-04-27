import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

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
    pandaframe = pandaframe.drop(['nbumps6', 'nbumps7', 'nbumps89'], axis=1)
    skewed = ['genergy', 'gpuls', 'energy', 'maxenergy']
    # replaces 0s with an extremely small float
    epsilon = np.finfo(float).eps
    for col in skewed:
        pandaframe[col] = np.log(x_train[col].replace(0, epsilon))

    normalizedinput = normalizer.fit_transform(x_train)


#done right before training
def normalizedata(x_train, x_test, y_train, y_test):
    x_train = x_train.drop(['nbumps6', 'nbumps7', 'nbumps89'], axis=1)
    x_test = x_test.drop(['nbumps6', 'nbumps7', 'nbumps89'], axis=1)
    skewed = ['genergy', 'gpuls', 'energy', 'maxenergy']
    # replaces 0s with an extremely small float
    epsilon = np.finfo(float).eps
    for col in skewed:
        x_train[col] = np.log(x_train[col].replace(0, epsilon))
        x_test[col] = np.log(x_test[col].replace(0, epsilon))

    numericals = ['seismic', 'seismoacoustic', 'ghazard', 'genergy', 'gpuls', 'gdenergy', 'gdpuls', 'ghazard',
                  'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'energy',
                  'maxenergy']

    normalizer = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numericals),
        ('catagoric', OneHotEncoder(sparse_output=False), ['shift'])
    ])

    x_train_norm = normalizer.fit_transform(x_train)
    x_test_norm = normalizer.transform(x_test)

    #smote = SMOTE(sampling_strategy=0.5,random_state=42)
    #x_train_full, y_train_full = smote.fit_resample(x_train_norm, y_train)

    #return x_train_full, x_test_norm, y_train_full.to_numpy(), y_test.to_numpy()
    return x_train_norm, x_test_norm, y_train.to_numpy(), y_test.to_numpy()

def split_train_test(df):
    x = df.drop('class', axis=1)
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return x_train, x_test, y_train, y_test

def cat_to_num(df):
    df['class'] = df['class'].map({b'1': 1, b'0': 0})

    hazard_map = {b'a': 0, b'b': 1, b'c': 2, b'd': 3}
    df['seismic'] = df['seismic'].map(hazard_map)
    df['seismoacoustic'] = df['seismoacoustic'].map(hazard_map)
    df['ghazard'] = df['ghazard'].map(hazard_map)
    df['shift'] = df['shift'].map({b'W': 0, b'N': 1})

def compute_weights(y_train):
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    return class_weight_dict
if name == "__main__"
    arff_file = arff.loadarff('./data/seismic-bumpsdata.arff')

    pandaframe = pd.DataFrame(arff_file[0])

    #rockframe = pandaframe[pandaframe['class'] == b'1']
    #plot(rockframe)

    x_train, x_test, y_train, y_test = prepareData(pandaframe)
    #print(x_test)

    #print(pandaframe)

    #getBursts(pandaframe)
    #plot(pandaframe)
