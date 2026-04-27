import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

def plot(pandaframe):
    pandaframe.hist(bins=50, figsize=(20,15))
    plt.savefig("attribute_histogram_plots")
    plt.show()


if __name__ == "__main__":

    arff_file = arff.loadarff('./data/seismic-bumpsdata.arff')

    pandaframe = pd.DataFrame(arff_file[0])

    #print(pandaframe.head())

    #print(pandaframe)

    #getBursts(pandaframe)
    plot(pandaframe)
    


    
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

