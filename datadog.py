import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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

#this is done right before training 
def normalizedata(x_train, x_test, y_train, y_test):
    numericals = ['seismic', 'seismoacoustic', 'ghazard', 'genergy', 'gpuls', 'gdenergy', 'gdpuls', 'ghazard', 
                  'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89', 'energy',
                  'maxenergy']
    normalizer = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numericals),
        ('catagoric', OneHotEncoder(sparse_output=False),['shift'])
    ])

    x_train_norm = normalizer.fit_transform(x_train)
    x_test_norm = normalizer.fit_transform(x_test)

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
    weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    return class_weight_dict

if __name__ == "__main__":

    arff_file = arff.loadarff('./data/seismic-bumpsdata.arff')

    pandaframe = pd.DataFrame(arff_file[0])

    x_train, x_test, y_train, y_test = prepareData(pandaframe)
    print(x_test)

    #print(pandaframe)

    #getBursts(pandaframe)
    #plot(pandaframe)


    
