import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_score(indicators, horizons, model, y_test, preds_test):
    # checking the results (RMSE value)
    rms=np.sqrt(np.mean(np.power((np.array(y_test[horizons])-preds_test),2)))
    print('\nRMSE using {} to Predict Price by {}'.format(indicators, model))
    print(rms)

def inverse_transform(horizons, scaler, y_train, y_test, preds_train, preds_test):
    train = y_train.copy()
    test = y_test.copy()
    train.loc[: ,horizons] = scaler.inverse_transform(y_train.loc[: ,horizons]).copy()
    test.loc[: ,horizons] = scaler.inverse_transform(y_test.loc[: ,horizons]).copy()
    preds_train = preds_train.reshape(y_train.shape[0], 1)
    preds_test = preds_test.reshape(y_test.shape[0], 1)
    preds_train = scaler.inverse_transform(preds_train)
    preds_test = scaler.inverse_transform(preds_test)
    return train, test, preds_train, preds_test

def plot_prediction(indicators, horizons, model, scaler, y_train, y_test, preds_train, preds_test):
    #plot
    y_train, y_test, preds_train, preds_test = inverse_transform(horizons, scaler, y_train, y_test, preds_train, preds_test)
    preds_train = pd.DataFrame(preds_train, columns = ['Predictions'], index = y_train.index)
    preds_test = pd.DataFrame(preds_test, columns = ['Predictions'], index = y_test.index)
    plt.plot(y_train[horizons], label='Train Price', color='g')
    plt.plot(y_test[horizons], label='Test Price', color='b')
    plt.plot(preds_train['Predictions'], label='{} Predicted Price'.format(model), color='r')
    plt.plot(preds_test['Predictions'], color='r')
    plt.title('Using {} to Predict Price by {}'.format(indicators, model))
    plt.legend()