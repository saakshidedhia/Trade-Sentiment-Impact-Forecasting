import pandas as pd
import os

# Load 4-hour block article data (with Time_Block, Interval_4h, Article_Count, embeddings)
articles = pd.read_excel("output/time_block_transformed.xlsx")

# Load hourly merged stock data for companies (with Timestamp and per-ticker Open/High/Low/Close/Volume prefixed columns)
stock = pd.read_excel("output/hourly_merged_stocks_for_companies.xlsx")

# Ensure Timestamp is datetime and drop timezone info if present
stock['Timestamp'] = pd.to_datetime(stock['Timestamp']).dt.tz_localize(None)

# Define 4-hour interval classifier
def classify_4h_interval(ts):
    h = ts.hour
    if h < 4:
        return '00:00–04:00'
    if h < 8:
        return '04:00–08:00'
    if h < 12:
        return '08:00–12:00'
    if h < 16:
        return '12:00–16:00'
    if h < 20:
        return '16:00–20:00'
    return '20:00–00:00'

# Assign block and date to stock data
stock['Interval_4h'] = stock['Timestamp'].apply(classify_4h_interval)
stock['Date'] = stock['Timestamp'].dt.date
stock['Time_Block'] = stock['Date'].astype(str) + ' ' + stock['Interval_4h']

# Identify numeric stock columns for aggregation
agg_map = {}
for col in stock.columns:
    lower = col.lower()
    if lower.endswith('_open'):
        agg_map[col] = 'first'
    elif lower.endswith('_high'):
        agg_map[col] = 'max'
    elif lower.endswith('_low'):
        agg_map[col] = 'min'
    elif lower.endswith('_close'):
        agg_map[col] = 'last'
    elif lower.endswith('_volume'):
        agg_map[col] = 'sum'

# Aggregate stock data into 4-hour blocks
agg_stock = stock.groupby('Time_Block').agg(agg_map).reset_index()

# Extract Date and Interval_4h back from Time_Block for clarity
split = agg_stock['Time_Block'].str.split(' ', n=1, expand=True)
agg_stock['Date'] = pd.to_datetime(split[0]).dt.date
agg_stock['Interval_4h'] = split[1]

# Compute block-level returns for each ticker
tickers = {col.rsplit('_',1)[0] for col in agg_map if col.lower().endswith('_open')}
for ticker in tickers:
    open_col = f"{ticker}_Open"
    close_col = f"{ticker}_Close"
    ret_col = f"{ticker}_Return"
    if open_col in agg_stock and close_col in agg_stock:
        agg_stock[ret_col] = (agg_stock[close_col] - agg_stock[open_col]) / agg_stock[open_col]

# Merge articles and aggregated stock on Time_Block
df_merged = pd.merge(
    articles,
    agg_stock,
    on='Time_Block',
    how='left',
    suffixes=('_art','_stk')
)

# Resolve Interval_4h and Date columns post-merge
# Preference to article side for Interval_4h and Date
if 'Interval_4h_art' in df_merged.columns:
    df_merged['Interval_4h'] = df_merged['Interval_4h_art']
    df_merged.drop(columns=['Interval_4h_art','Interval_4h_stk'], inplace=True)
else:
    df_merged['Interval_4h'] = df_merged['Interval_4h']

if 'Date_art' in df_merged.columns:
    df_merged['Date'] = df_merged['Date_art']
    df_merged.drop(columns=['Date_art','Date_stk'], inplace=True)

# Optional: sort by Date and Interval_4h ordering
order = ['00:00–04:00','04:00–08:00','08:00–12:00','12:00–16:00','16:00–20:00','20:00–00:00']
df_merged['Interval_4h'] = pd.Categorical(df_merged['Interval_4h'], categories=order, ordered=True)
df_merged = df_merged.sort_values(['Date','Interval_4h'])

# Save merged dataset
os.makedirs('output', exist_ok=True)
df_merged.to_excel('output/company_4h_merged.xlsx', index=False)
print("✅ company_4h_merged.xlsx created in output/")
