import os

BUMPS_PATH = os.path.join("datasets", "bumps")

import pandas as pd
from scipy.io import arff

def load_bumps_data(bumps_path=BUMPS_PATH):
    arff_path = os.path.join(bumps_path, "seismic_bumps_clean.arff")
    arff_file = arff.loadarff(arff_path)
    return pd.DataFrame(arff_file[0])
    
bumps = load_bumps_data()
bumps.head()

bumps.info()

bumps["seismic"].value_counts()

bumps["seismoacoustic"].value_counts()

bumps["class"].value_counts()

bumps.describe()

%matplotlib inline
import matplotlib.pyplot as plt
bumps.hist(bins=50, figsize=(20,15))
plt.show()

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    
train_set, test_set = split_train_test(bumps, 0.2)
len(train_set)

len(test_set)

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

bumps_with_id = bumps.reset_index()
train_set, test_set = split_train_test_by_id(bumps_with_id, 0.2, "index")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(bumps, test_size=0.2, random_state=42)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(bumps, bumps["class"]):
    strat_train_set = bumps.loc[train_index]
    strat_test_set = bumps.loc[test_index]
    
strat_test_set["class"].value_counts()

bumps.plot(kind="scatter", x="gpuls", y="ghazard")

corr_matrix = bumps.corr(numeric_only=True)

from pandas.plotting import scatter_matrix

attributes = ["genergy", "energy", "gpuls", "ghazard"]
scatter_matrix(bumps[attributes], figsize=(12, 8))

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
bumps_num = bumps.drop(columns=["seismic", "seismoacoustic", "ghazard", "shift"])
imputer.fit(bumps_num)

imputer.statistics_

X = imputer.transform(bumps_num)

bumps_cat = bumps[["seismic", "seismoacoustic", "shift", "ghazard"]]
bumps_cat.head(10)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
bumps_cat_1hot = cat_encoder.fit_transform(bumps_cat)
bumps_cat_1hot

cat_encoder.categories_

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('minmax_scaler', MinMaxScaler()),
])

bumps_num_tr = num_pipeline.fit_transform(bumps_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(bumps_num)
cat_attribs = ["seismic", "seismoacoustic", "shift", "ghazard"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

bumps_prepared = full_pipeline.fit_transform(bumps)