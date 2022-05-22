import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

# Get the symbols of S&P 100
tickers = pd.read_csv('../data/sp100tickers.csv')

# Get all stock data of all S&P 100
stocks = []
for symbol, sector in tickers.values:
    stock = pd.read_csv('../data/technical data/{}.csv'.format(symbol), thousands=',')
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock['Symbol'] = symbol
    stock['Sector'] = sector
    stock.set_index(['Date'], inplace=True)
    stocks.append(stock)

# Plot the stock ticker of AAPL company
apple = stocks[0].loc[(stocks[0]['Symbol'] == "AAPL") & (stocks[0]['Sector'] == "Information Technology")]
plt.figure(figsize=(16,8))
plt.plot(apple.index, apple['Close'], label='Close Price history')
plt.legend()
plt.show()

# Compute different techinical indicators data
for stock in stocks:
    # Moving Average Convergence Divergence (MACD) with simple moving average method and 26 days & 12 days for the slow and fast periods respectively.
    stock.ta.macd(close='close', fast=12, slow=26, append=True)
    # Money Flow Index (MFI) over a period of 14 days.
    stock.ta.mfi(high='high', low='low', close='close', volume='volume', length=14, append=True)
    # FastK and FastD values of Stochastic Oscillator using 14,3, and 3 days for FastK, FastD, SlowD respectively.
    stock.ta.stoch(high='high', low='low', close='close', k=14, d=3, smooth_k=3, append=True)
    # Relative Strenght Index (RSI) using 14 days and weighted moving average.
    stock.ta.rsi(close='close', length=14, scalar=100, drift=1, append=True)
    # Relative Strenght Index (RSI) using 14 days and weighted moving average.
    stock.ta.ad(high='high', low='low', close='close', volume='volume', length=14, scalar=100, drift=1, append=True)
    # Price Rate of Change (ROC) over 252 or 126 trading days.
    stock.ta.roc(close='close', length=126, append=True)
    # Fill the column backward, that is, the column value of Nan, and fill it with the latter of its columns
    stock.fillna(method = 'backfill', axis = 0, inplace =True)

# Daily MSCI industry index prices (MACD, 252 days, 12 days)
# Daily MSCI industry index prices (MACD, 26 days, 12 days)
names = ['Information Technology', 'Health Care','Financials', 'Consumer Discretionary', 'Communication Services', 'Industrials', 'Consumer Staples', 'Energy', 'Utilities', 'Real Estate', 'Materials']
industries_macd = {}
for name in names:
    industry = pd.read_csv('../data/fundamental data/Daily MSCI industry index prices/S&P 500 {} Historical Data.csv'.format(name), thousands=',', index_col=0, parse_dates=True)
    industry_macd_12_252 = industry.ta.macd(close='Price', fast=12, slow=252)
    industries_macd['{}_industry_macd_12_252'.format(name)] = industry_macd_12_252
    industry_macd_12_26 = industry.ta.macd(close='Price', fast=12, slow=26)
    industries_macd['{}_industry_macd_12_26'.format(name)] = industry_macd_12_26
# S&P 500 Futures prices (MACD, 252 days, 12 days)
sp500_future = pd.read_csv('../data/fundamental data/S&P 500 Futures Historical Data.csv', thousands=',', index_col=0, parse_dates=True)
sp500_future_macd = sp500_future.ta.macd(close='Price', fast=12, slow=252)
# Daily Trade Weighted U.S. Dollar Index against Major Currencies (MACD, 252 days, 12 days)
dollar_index = pd.read_csv('../data/fundamental data/Daily Trade Weighted U.S. Dollar Index.csv', thousands=',', index_col=0, parse_dates=True)
dollar_index_macd = dollar_index.ta.macd(close='DTWEXAFEGS', fast=12, slow=252)
# 10 year to 2 year constant maturity rate (MACD, 26 days, 12 days)
constant_maturity = pd.read_csv(
    '../data/fundamental data/10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity.csv', thousands=',', index_col=0, parse_dates=True)
constant_maturity_macd = constant_maturity.ta.macd(close='T10Y2Y', fast=12, slow=26)

# Eliminate data in mismatched dates and fill the Nan value
for key, macd in industries_macd.items():
    macd = macd[macd.index.isin(apple.index)].fillna(method = 'backfill', axis = 0)
    industries_macd[key] = macd

# Eliminate data in mismatched dates and fill the Nan value
sp500_future_macd = sp500_future_macd[sp500_future_macd.index.isin(apple.index)].fillna(method = 'backfill', axis = 0)
dollar_index_macd = dollar_index_macd[dollar_index_macd.index.isin(apple.index)].fillna(method = 'backfill', axis = 0)
constant_maturity_macd = constant_maturity_macd[constant_maturity_macd.index.isin(apple.index)].fillna(method = 'backfill', axis = 0)

# Create dataframe with all target variable
x_all = pd.DataFrame()
y_all = pd.DataFrame()
for stock in stocks:
    # Get the responding sector macd
    industry = stock['Sector'][0]
    industry_macd_12_252 = industries_macd['{}_industry_macd_12_252'.format(industry)]
    industry_macd_12_26 = industries_macd['{}_industry_macd_12_26'.format(industry)]
    # Create dataframe with the target variable
    x = pd.DataFrame({'Symbol': stock['Symbol'],
                      'Sector': stock['Sector'],
                      'Close': stock['Close'],
                      'MACD_12_26_9': stock['MACD_12_26_9'],
                      'MACDh_12_26_9': stock['MACDh_12_26_9'],
                      'MACDs_12_26_9': stock['MACDs_12_26_9'],
                      'MFI_14': stock['MFI_14'],
                      'STOCHk_14_3_3': stock['STOCHk_14_3_3'],
                      'STOCHd_14_3_3': stock['STOCHd_14_3_3'],
                      'RSI_14': stock['RSI_14'],
                      'AD': stock['AD'],
                      'ROC_126': stock['ROC_126'],
                      'industry_macd_12_252': industry_macd_12_252['MACD_12_252_9'],
                      'industry_macdh_12_252': industry_macd_12_252['MACDh_12_252_9'],
                      'industry_macds_12_252': industry_macd_12_252['MACDs_12_252_9'],
                      'industry_macd_12_26': industry_macd_12_26['MACD_12_26_9'],
                      'industry_macdh_12_26': industry_macd_12_26['MACDh_12_26_9'],
                      'industry_macds_12_26': industry_macd_12_26['MACDs_12_26_9'],
                      'sp500_future_macd_12_252': sp500_future_macd['MACD_12_252_9'],
                      'sp500_future_macdh_12_252': sp500_future_macd['MACDh_12_252_9'],
                      'sp500_future_macds_12_252': sp500_future_macd['MACDs_12_252_9'],
                      'dollar_index_macd_12_252': dollar_index_macd['MACD_12_252_9'],
                      'dollar_index_macdh_12_252': dollar_index_macd['MACDh_12_252_9'],
                      'dollar_index_macds_12_252': dollar_index_macd['MACDs_12_252_9'],
                      'constant_maturity_macd_12_26': constant_maturity_macd['MACD_12_26_9'],
                      'constant_maturity_macdh_12_26': constant_maturity_macd['MACDh_12_26_9'],
                      'constant_maturity_macds_12_26': constant_maturity_macd['MACDs_12_26_9']})
    # Add the actual close price after 1 day
    y = pd.DataFrame({'Symbol': stock['Symbol'],
                      'Sector': stock['Sector'],
                      'Close_after_1_day': stock['Close'].shift(-1)})
    # Fill the value of the last day without the actual price data
    y = y.fillna(method='pad', axis=0)
    x_all = x_all.append(x)
    y_all = y_all.append(y)

# NOTE: While splitting the data into train and test set, we cannot use random splitting since that will destroy the time component.
# So here we have set the last year’s data into test set and the 4 years’ data before that into train set.

# Split into train and validation
x_train = pd.DataFrame()
x_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
for symbol,sector in tickers.values:
    x_train = x_train.append(x_all.loc[(x_all['Symbol'] == symbol) & (x_all['Sector'] == sector)][:1007])
    x_test = x_test.append(x_all.loc[(x_all['Symbol'] == symbol) & (x_all['Sector'] == sector)][1007:])
    y_train = y_train.append(y_all.loc[(x_all['Symbol'] == symbol) & (x_all['Sector'] == sector)][:1007])
    y_test = y_test.append(y_all.loc[(x_all['Symbol'] == symbol) & (x_all['Sector'] == sector)][1007:])

print("x_all.shape: " + str(x_all.shape))
print("y_all.shape: " + str(y_all.shape))
print("x_train.shape: " + str(x_train.shape))
print("x_test.shape: " + str(x_test.shape))
print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))

x_all.to_csv('../data/x_all.csv')
y_all.to_csv('../data/y_all.csv')
x_train.to_csv('../data/x_train.csv')
x_test.to_csv('../data/x_test.csv')
y_train.to_csv('../data/y_train.csv')
y_test.to_csv('../data/y_test.csv')




