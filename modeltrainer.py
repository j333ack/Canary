from datadog import *
import tensorflow as tf
from tensorflow import keras


def main():
    arff_file = arff.loadarff('./data/seismic-bumpsdata.arff')
    pandaframe = pd.DataFrame(arff_file[0])
    x_train, x_test, y_train, y_test = prepareData(pandaframe)
    class_weights_dict = compute_weights(y_train)

    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(x_train.shape[1],)))

    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(16, activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(x_test, y_test),
        class_weight=class_weight_dict

    )

    model.evaluate(x_test,y_test)

main()



