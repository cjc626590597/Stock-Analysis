import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.svm import SVR 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from sklearn.svm import SVR
def svr_train(indicators, horizons, X_train, y_train, X_test, y_test, model=None):
    if model == None:
        model = SVR(C=0.001, kernel='poly', gamma=1, max_iter=1000000)
    model.fit(X_train[indicators], np.array(y_train[horizons]).reshape(-1))
    preds_train = model.predict(X_train[indicators])
    preds_test = model.predict(X_test[indicators])
    return preds_train, preds_test


# Initialising the LSTM
def build_model(indicators, units=32, activation='tanh', learning_rate=0.001, dropout_rate=0.1):
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=units, activation=activation, return_sequences=True, input_shape=(60, len(indicators))))
    model.add(Dropout(dropout_rate))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=units, activation=activation, return_sequences=True))
    model.add(Dropout(dropout_rate))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=units, activation=activation))
    model.add(Dropout(dropout_rate))

    # Adding the output layer
    model.add(Dense(units=1, activation=activation))

    # Compiling the LSTM
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def transform_60(indicators, X_data, y_data, horizons):
    # Creating a data structure with 60 timesteps and 1 output
    x = []
    y = []
    for i in range(59, X_data.shape[0]):
        x.append(np.array(X_data.iloc[i-59:i+1][indicators]))
        y.append(np.array(y_data.iloc[i][horizons][0]))
    x, y = np.array(x), np.array(y)
    return x, y


def lstm_train(indicators, horizons, X_train, y_train, X_test, y_test, model=None):
    x = X_train.iloc[-59:]
    y = y_train.iloc[-59:]
    X_test = pd.concat([x, X_test], axis=0)
    y_test = pd.concat([y, y_test], axis=0)

    X_train, y_train = transform_60(indicators, X_train, y_train, horizons)
    X_test, y_test = transform_60(indicators, X_test, y_test, horizons)

    if model == None:
        model = build_model(indicators)
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    return preds_train, preds_test

# Compute RMSE
def score(horizons, y_test, preds_test):
    return np.sqrt(np.mean(np.power((np.array(y_test[horizons]).ravel()-preds_test.ravel()),2)))