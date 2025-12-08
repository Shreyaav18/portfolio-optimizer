import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def calculate_technical_indicators(df):
    rsi = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'])
    bb = BollingerBands(df['Close'], window=20)
    
    df['RSI'] = rsi
    df['MACD'] = macd.macd()
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    
    return df

def prepare_data(tickers, start_date, end_date):
    raw_data = download_stock_data(tickers, start_date, end_date)
    
    processed_dfs = []
    
    for ticker in tickers:
        if len(tickers) > 1:
            stock_data = raw_data[ticker].copy()
        else:
            stock_data = raw_data.copy()
        
        stock_data = calculate_technical_indicators(stock_data)
        
        stock_data = stock_data[['Close', 'RSI', 'MACD', 'BB_high', 'BB_low']].copy()
        stock_data.columns = [f'{ticker}_{col}' for col in stock_data.columns]
        
        processed_dfs.append(stock_data)
    
    combined_df = pd.concat(processed_dfs, axis=1)
    combined_df = combined_df.dropna()
    
    for col in combined_df.columns:
        combined_df[col] = (combined_df[col] - combined_df[col].mean()) / combined_df[col].std()
    
    return combined_df

def split_data(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df