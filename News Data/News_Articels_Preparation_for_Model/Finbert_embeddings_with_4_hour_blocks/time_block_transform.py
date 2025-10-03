import pandas as pd
import itertools

# Load original Excel
df = pd.read_excel("output/articles_with_finbert.xlsx")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Classify each timestamp into 4-hour interval
def classify_4h_interval(ts):
    hour = ts.hour
    if 0 <= hour < 4:
        return "00:00–04:00"
    elif 4 <= hour < 8:
        return "04:00–08:00"
    elif 8 <= hour < 12:
        return "08:00–12:00"
    elif 12 <= hour < 16:
        return "12:00–16:00"
    elif 16 <= hour < 20:
        return "16:00–20:00"
    else:
        return "20:00–00:00"

df["Interval_4h"] = df["Timestamp"].apply(classify_4h_interval)
df["Date"] = df["Timestamp"].dt.date
df["Time_Block"] = df["Date"].astype(str) + " " + df["Interval_4h"]

# Build all possible 4-hour time blocks
all_dates = df["Date"].unique()
intervals = ["00:00–04:00", "04:00–08:00", "08:00–12:00", "12:00–16:00", "16:00–20:00", "20:00–00:00"]
all_blocks = pd.DataFrame(list(itertools.product(all_dates, intervals)), columns=["Date", "Interval_4h"])
all_blocks["Time_Block"] = all_blocks["Date"].astype(str) + " " + all_blocks["Interval_4h"]

# Count articles in each block
block_counts = df["Time_Block"].value_counts().reset_index()
block_counts.columns = ["Time_Block", "Article_Count"]
block_full = pd.merge(all_blocks, block_counts, on="Time_Block", how="left").fillna({"Article_Count": 0})
block_full["Article_Count"] = block_full["Article_Count"].astype(int)

# Merge full blocks into df
df = pd.merge(df, block_full, on="Time_Block", how="right")
df.rename(columns={"Interval_4h_y": "Interval_4h", "Date_y": "Date"}, inplace=True)
df.drop(columns=["Interval_4h_x", "Date_x"], inplace=True)

# Create Time_Block_Count (enumerated)
time_block_map = {tb: i for i, tb in enumerate(sorted(block_full["Time_Block"].unique()), start=1)}
df["Time_Block_Count"] = df["Time_Block"].map(time_block_map)

# Convert count columns to int
df["Article_Count"] = df["Article_Count"].astype(int)
if "Article_Number" in df.columns:
    df["Article_Number"] = df["Article_Number"].fillna(0).astype(int)

# Reorder columns
first_cols = ["Time_Block_Count", "Interval_4h", "Article_Count"]
remaining_cols = [col for col in df.columns if col not in first_cols]
df = df[first_cols + remaining_cols]
# Save the final transformed DataFrame to Excel
import os

# Make sure the 'output' folder exists
os.makedirs("output", exist_ok=True)

# Save as Excel
df.to_excel("output/time_block_transformed.xlsx", index=False)

print("✅ File saved to output/time_block_transformed.xlsx")
