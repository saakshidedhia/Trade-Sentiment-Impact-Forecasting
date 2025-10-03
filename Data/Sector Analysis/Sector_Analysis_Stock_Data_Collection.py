import pandas as pd
import yfinance as yf

# 1. Define sector ETF tickers
sector_etfs = {
    "Financials": "XLF",
    "Technology": "XLK",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Consumer Staples": "XLP",
    "Communication Services": "XLC"
}

# 2. Download Adjusted Close Prices
tickers = list(sector_etfs.values())
data = yf.download(tickers, start="2025-01-01", end="2025-04-24", group_by='ticker', auto_adjust=True)

# 3. Create panel dataset: Date × Sector × Return
panel_data = []

for sector, ticker in sector_etfs.items():
    sector_prices = data[ticker]["Close"].dropna()
    sector_returns = sector_prices.pct_change().dropna()
    temp_df = pd.DataFrame({
        "Date": sector_returns.index,
        "Sector": sector,
        "Return": sector_returns.values
    })
    panel_data.append(temp_df)

# 4. Combine all sector data into one DataFrame
panel_df = pd.concat(panel_data).reset_index(drop=True)

# 5. View sample
print(panel_df.head())

# 6. Save the panel dataframe to Excel
import os

# Create an 'output' folder if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save to Excel
panel_df.to_excel('output/sector_panel_returns.xlsx', index=False)

print("✅ Sector panel dataset saved to output/sector_panel_returns.xlsx")
