import pandas as pd
import numpy as np
import os


STOCK_FEATURE_COLUMNS = ['DlyPrc', 'DlyCap', 'DlyPrevPrc', 'DlyPrevCap', 'DlyRet', 'DlyRetx', 'DlyRetI', 'DlyOrdDivAmt', 'DlyNonOrdDivAmt', 'DlyFacPrc', 'DlyClose', 'DlyLow', 'DlyHigh', 'DlyBid', 'DlyAsk', 'DlyOpen', 'DlyPrcVol', 'vwretd', 'vwretx', 'ewretd', 'ewretx', 'sprtrn']


def scale(v,axis = 0):
    '''
    scipykit scaler doesn't support 3D array
    '''
    return (v - v.min(axis=axis))/(v.max(axis=axis) - v.min(axis=axis)+0.001)

def get_stock_data(data, tickers, date):
    single_day_data = data.loc[data['YYYYMMDD'] == date]
    feature = []
    for ticker in tickers:
        tem = single_day_data.loc[single_day_data['Ticker'] == ticker][STOCK_FEATURE_COLUMNS].values
        feature.append(tem)
    feature = np.concatenate(feature)
    return feature

def get_tag_data(data, tickers, date_one, date_two):
    day_one_data = data.loc[data['YYYYMMDD'] == date_one]
    day_two_data = data.loc[data['YYYYMMDD'] == date_two]
    tag = []
    for ticker in tickers:
        tem_1 = day_one_data.loc[day_one_data['Ticker'] == ticker]['DlyClose'].values
        tem_2 = day_two_data.loc[day_two_data['Ticker'] == ticker]['DlyClose'].values
        tem = (tem_2-tem_1)/tem_1
        tag.append(int(tem>0.01)-int(tem<-0.01) + 1)
    tag = np.stack(tag)
    return tag

def get_price_data(data, tickers, date_one):
    day_one_data = data.loc[data['YYYYMMDD'] == date_one]
    prices = []
    for ticker in tickers:
        price = day_one_data.loc[day_one_data['Ticker'] == ticker]['DlyClose'].values
        prices.append(price)
    prices = np.stack(prices)
    return prices

def get_stock_price(tickers,PATH_TO_DATA_FOLDER):
    path_to_csv = os.path.join(PATH_TO_DATA_FOLDER, "wrds.csv")
    print(path_to_csv)
    wrds = pd.read_csv(path_to_csv)
    
    dates = list(set(wrds['YYYYMMDD'].values))
    dates = sorted(dates)
    
    price_per_date_per_company =  np.stack([get_price_data(wrds, tickers,date) for date in dates])
    return price_per_date_per_company

def get_dates(tickers,PATH_TO_DATA_FOLDER):
    path_to_csv = os.path.join(PATH_TO_DATA_FOLDER, "wrds.csv")
    print(path_to_csv)
    wrds = pd.read_csv(path_to_csv)
    
    dates = list(set(wrds['YYYYMMDD'].values))
    dates = sorted(dates)

    return dates

def get_stock_feature_tag(tickers,PATH_TO_DATA_FOLDER):
    path_to_csv = os.path.join(PATH_TO_DATA_FOLDER, "wrds.csv")
    wrds = pd.read_csv(path_to_csv)
    
    dates = list(set(wrds['YYYYMMDD'].values))
    dates = sorted(dates)
    
    data_per_date_per_company = np.stack([get_stock_data(wrds,tickers,date) for date in dates[:-1]])
    data_per_date_per_company = scale(data_per_date_per_company, axis = 0)
    
    date_zip = list(zip(dates[:-1],dates[1:]))
    tag_per_date_per_company =  np.stack([get_tag_data(wrds, tickers,date_one,date_two) for date_one,date_two in date_zip])
    return data_per_date_per_company,tag_per_date_per_company

def get_stock_feature(tickers,PATH_TO_DATA_FOLDER):
    path_to_csv = os.path.join(PATH_TO_DATA_FOLDER, "wrds.csv")
    wrds = pd.read_csv(path_to_csv)
    
    dates = list(set(wrds['YYYYMMDD'].values))
    dates = sorted(dates)
    
    data_per_date_per_company = np.stack([get_stock_data(wrds,tickers,date) for date in dates[:-1]])
    data_per_date_per_company = scale(data_per_date_per_company, axis = 0)
    
    return data_per_date_per_company
    
def get_stock_feature_dates(tickers, PATH_TO_DATA_FOLDER, dates_file = 'train_dates.csv'):
    path_to_csv = os.path.join(PATH_TO_DATA_FOLDER, "wrds.csv")
    path_to_dates = os.path.join(PATH_TO_DATA_FOLDER, dates_file)
    
    wrds = pd.read_csv(path_to_csv)
    
    dates = pd.read_csv(path_to_dates)
    dates = [np.int64(x.replace('-','')) for x in dates['trading_day'].values]
    
    data_per_date_per_company = np.stack([get_stock_data(wrds,tickers,date) for date in dates[:-1]])
    data_per_date_per_company = scale(data_per_date_per_company, axis = 0)
    
    date_zip = list(zip(dates[:-1],dates[1:]))
    tag_per_date_per_company =  np.stack([get_tag_data(wrds, tickers,date_one,date_two) for date_one,date_two in date_zip])
    return data_per_date_per_company,tag_per_date_per_company
    
    