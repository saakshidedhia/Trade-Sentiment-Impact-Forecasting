from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time

api_key = 'G0LIEZ1KL46PL3BR'
ts = TimeSeries(key=api_key, output_format='pandas')

tickers = ['AAPL', 'TSLA', 'INTC', 'AMD','NVDA','F','GM','WHR','NKE','CAT','BA','BABA', 'JD','PDD']
all_data = []

for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    try:
        data, _ = ts.get_intraday(symbol=ticker, interval='60min', outputsize='full')
        data = data.rename(columns={
            '1. open': f'{ticker}_open',
            '2. high': f'{ticker}_high',
            '3. low': f'{ticker}_low',
            '4. close': f'{ticker}_close',
            '5. volume': f'{ticker}_volume'
        })
        data.index.name = 'Timestamp'
        all_data.append(data)
    except:
        print(f"❌ Could not fetch {ticker}")
    time.sleep(15)  # Respect rate limit

# Merge on timestamp
from functools import reduce
merged = reduce(lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'), all_data)
merged = merged.sort_index()

# Save to Excel
merged.to_excel("output/hourly_merged_stocks_for_companies.xlsx")
print("✅ Merged file saved as output/hourly_merged_stocks.xlsx")
