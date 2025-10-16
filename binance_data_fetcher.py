#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance OHLCV Data Fetcher for ATLAS Apex S5-F1.1-soft-k2 Backtest
Downloads historical 15m candles for specified symbols
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone
import pytz

def fetch_binance_klines(symbol, interval="15m", start_time=None, end_time=None, limit=1000):
    """Fetch klines from Binance API"""
    base_url = "https://api.binance.com/api/v3/klines"
    
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time: params["startTime"] = int(start_time)
    if end_time: params["endTime"] = int(end_time)
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def download_historical_data(symbol, interval="15m", start_date="2024-03-01", end_date="2024-10-16", timezone_str="Europe/Istanbul"):
    """Download full historical dataset with pagination"""
    print(f"Fetching {symbol} {interval} data from {start_date} to {end_date}")
    
    tz = pytz.timezone(timezone_str)
    start_dt = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    end_dt = tz.localize(datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S"))
    
    start_ts = int(start_dt.astimezone(pytz.UTC).timestamp() * 1000)
    end_ts = int(end_dt.astimezone(pytz.UTC).timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    
    while current_start < end_ts:
        print(f"  Fetching from {datetime.fromtimestamp(current_start/1000, tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC...")
        
        klines = fetch_binance_klines(symbol=symbol, interval=interval, start_time=current_start, end_time=end_ts, limit=1000)
        
        if not klines:
            print("  No more data received")
            break
        
        all_klines.extend(klines)
        current_start = klines[-1][6] + 1
        time.sleep(0.2)
        
        if klines[-1][6] >= end_ts:
            break
    
    print(f"  Total candles fetched: {len(all_klines)}")
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    return df

def main():
    """Main execution: Download data for all required symbols"""
    symbols = ["BTCUSDT", "SOLUSDT", "ETHUSDT", "AVAXUSDT"]
    interval = "15m"
    start_date = "2024-03-01"
    end_date = "2024-10-16"
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Downloading {symbol}")
        print('='*60)
        
        df = download_historical_data(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, timezone_str="Europe/Istanbul")
        
        if df.empty:
            print(f"  WARNING: No data downloaded for {symbol}")
            continue
        
        filename = f"{symbol}_{interval}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n  ✅ Saved: {filename}")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"  Expected blocks (3200 bars): {len(df) // 3200}")
        print("\n  Sample data (first 3 rows):")
        print(df.head(3).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("✅ All downloads complete!")
    print('='*60)

if __name__ == "__main__":
    main()