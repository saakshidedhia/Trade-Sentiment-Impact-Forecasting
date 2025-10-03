import yfinance as yf
import pandas as pd
from functools import reduce
import os

# Define world indices to fetch (Yahoo Finance tickers)
indices = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'Dow Jones': '^DJI',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'CAC 40': '^FCHI',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI',
    'Shanghai Composite': '000001.SS'
}

all_data = []

for name, ticker in indices.items():
    print(f"Fetching hourly data for {name} ({ticker})...")
    try:
        data = yf.download(ticker, start="2024-03-28", end="2024-04-24", interval="1d")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.columns = [f"{name}_{col}" for col in data.columns]  # Add index name prefix
        data.index.name = 'Timestamp'
        data.index = data.index.tz_localize(None)  # Remove timezone for Excel compatibility
        all_data.append(data)
    except Exception as e:
        print(f"❌ Failed to fetch {name}: {e}")

# Merge all indices on timestamp
merged = reduce(lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'), all_data)
merged = merged.sort_index()

# Save to Excel
os.makedirs("output", exist_ok=True)
merged.to_excel("output/daily_world_indices.xlsx")
print("✅ World indices saved to output/hourly_world_indices.xlsx")
