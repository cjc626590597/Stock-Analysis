import pandas as pd

def load_stickers():
    tickers = pd.read_csv('../data/sp100tickers.csv')
    return tickers

def load_dataset():
    X_train = pd.read_csv('../data/X_train.csv', thousands=',', index_col=0, parse_dates=True)
    X_test = pd.read_csv('../data/X_test.csv', thousands=',', index_col=0, parse_dates=True)
    y_train = pd.read_csv('../data/y_train.csv', thousands=',', index_col=0, parse_dates=True)
    y_test = pd.read_csv('../data/y_test.csv', thousands=',', index_col=0, parse_dates=True)
    return X_train, X_test, y_train, y_test

def load_dataset_7():
    X_train_7 = pd.read_csv('../data/X_train_7.csv', thousands=',', index_col=0, parse_dates=True)
    X_test_7 = pd.read_csv('../data/X_test_7.csv', thousands=',', index_col=0, parse_dates=True)
    y_train_7 = pd.read_csv('../data/y_train_7.csv', thousands=',', index_col=0, parse_dates=True)
    y_test_7 = pd.read_csv('../data/y_test_7.csv', thousands=',', index_col=0, parse_dates=True)
    return X_train_7, X_test_7, y_train_7, y_test_7

def load_indicators3():
    technical_indicators = ['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'MFI_14', 'STOCHk_14_3_3',
                            'STOCHd_14_3_3', 'RSI_14', 'AD', 'ROC_126']
    fundamental_indicators = ['Close', 'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                              'industry_macd_12_26', 'industry_macdh_12_26', 'industry_macds_12_26',
                              'sp500_future_macd_12_252', 'sp500_future_macdh_12_252', 'sp500_future_macds_12_252',
                              'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252',
                              'constant_maturity_macd_12_26', 'constant_maturity_macdh_12_26',
                              'constant_maturity_macds_12_26']
    combined_indicators = ['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'MFI_14', 'STOCHk_14_3_3',
                             'STOCHd_14_3_3', 'RSI_14', 'AD', 'ROC_126',
                             'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                             'industry_macd_12_26', 'industry_macdh_12_26', 'industry_macds_12_26',
                             'sp500_future_macd_12_252', 'sp500_future_macdh_12_252', 'sp500_future_macds_12_252',
                             'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252',
                             'constant_maturity_macd_12_26', 'constant_maturity_macdh_12_26',
                             'constant_maturity_macds_12_26']
    return technical_indicators, fundamental_indicators, combined_indicators

def load_indicators5():
    all_indicators = ['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'MFI_14', 'STOCHk_14_3_3',
                      'STOCHd_14_3_3', 'RSI_14', 'AD', 'ROC_126',
                      'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                      'industry_macd_12_26', 'industry_macdh_12_26', 'industry_macds_12_26',
                      'sp500_future_macd_12_252', 'sp500_future_macdh_12_252', 'sp500_future_macds_12_252',
                      'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252',
                      'constant_maturity_macd_12_26', 'constant_maturity_macdh_12_26', 'constant_maturity_macds_12_26']
    combined_indicators1 = ['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'STOCHk_14_3_3',
                              'STOCHd_14_3_3', 'AD', 'ROC_126',
                              'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252']
    combined_indicators2 = ['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'STOCHk_14_3_3', 'AD',
                              'ROC_126',
                              'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                              'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252']
    combined_indicators3 = ['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'AD', 'ROC_126',
                              'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                              'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252',
                              'industry_macd_12_26', 'industry_macdh_12_26', 'industry_macds_12_26']
    combined_indicators4 = ['Close', 'AD', 'ROC_126',
                              'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                              'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252',
                              'constant_maturity_macd_12_26', 'constant_maturity_macdh_12_26',
                              'constant_maturity_macds_12_26',
                              'industry_macd_12_26', 'industry_macdh_12_26', 'industry_macds_12_26']
    combined_indicators5 = ['Close', 'AD',
                              'industry_macd_12_252', 'industry_macdh_12_252', 'industry_macds_12_252',
                              'dollar_index_macd_12_252', 'dollar_index_macdh_12_252', 'dollar_index_macds_12_252',
                              'constant_maturity_macd_12_26', 'constant_maturity_macdh_12_26',
                              'constant_maturity_macds_12_26',
                              'sp500_future_macd_12_252', 'sp500_future_macdh_12_252', 'sp500_future_macds_12_252',
                              'industry_macd_12_26', 'industry_macdh_12_26', 'industry_macds_12_26']
    return all_indicators, combined_indicators1, combined_indicators2, combined_indicators3, combined_indicators4, combined_indicators5

def load_horizons():
    horizons = ['Close_after_1_day']
    horizons_7 = ['Close_after_7_day']
    return horizons, horizons_7