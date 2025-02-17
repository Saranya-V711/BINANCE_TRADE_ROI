import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("TRADES_ROI.csv")

# overview of the data
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Understand the structure and content
print(df.info())
print(df.describe())
print(df['Trade_History'].head())

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# Drop rows where either Port_IDs or Trade_History is missing
df.dropna(subset=['Port_IDs', 'Trade_History'], inplace=True)

print(df.isna().sum())
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

import ast

# Sample data with single quotes
data = "{'key': 'value'}"

# Safely evaluate the string
parsed_data = ast.literal_eval(data)
print(parsed_data)

def parse_trade_history(trade_history_str):
    try:
        # Safely evaluate the string to a Python object
        return ast.literal_eval(trade_history_str)
    except (ValueError, SyntaxError):
        # Return an empty list if parsing fails
        return []

# Apply the function to create a new column with parsed data
df['Parsed_Trade_History'] = df['Trade_History'].apply(parse_trade_history)

# Explode the parsed trade history into separate rows
df_exploded = df.explode('Parsed_Trade_History').reset_index(drop=True)

# Drop rows where Parsed_Trade_History is empty
df_exploded = df_exploded[df_exploded['Parsed_Trade_History'].notna()]

# Normalize the nested dictionaries into columns
trade_details = pd.json_normalize(df_exploded['Parsed_Trade_History'])

# Combine with the Port_IDs
trade_details['Port_ID'] = df_exploded['Port_IDs']

# Display the resulting DataFrame
print(trade_details.head())


# Convert timestamp (assuming column is named 'time' and is in milliseconds)
if 'time' in trade_details.columns:
    trade_details['time'] = pd.to_datetime(trade_details['time'], unit='ms')

# Convert numeric columns to proper numeric types
for col in ['price', 'quantity', 'realizedProfit']:
    if col in trade_details.columns:
        trade_details[col] = pd.to_numeric(trade_details[col], errors='coerce')

# Optionally, drop rows with NaN values in critical numeric columns
trade_details.dropna(subset=['price', 'quantity', 'realizedProfit'], inplace=True)

print(trade_details.info())

# Define a function to classify trades
def classify_trade(row):
    if row['side'] == 'BUY' and row.get('positionSide', 'LONG') == 'LONG':
        return 'long_open'
    elif row['side'] == 'SELL' and row.get('positionSide', 'LONG') == 'LONG':
        return 'long_close'
    elif row['side'] == 'SELL' and row.get('positionSide', 'SHORT') == 'SHORT':
        return 'short_open'
    elif row['side'] == 'BUY' and row.get('positionSide', 'SHORT') == 'SHORT':
        return 'short_close'
    else:
        return 'unknown'

# Apply the function to classify trades
trade_details['PositionAction'] = trade_details.apply(classify_trade, axis=1)

# Display the classified trades
print(trade_details[['Port_ID', 'time', 'symbol', 'PositionAction']].head())

# For instance, if you want to compute cumulative profit for each account:
trade_details.sort_values(by='time', inplace=True)
trade_details['cumulative_pnl'] = trade_details.groupby('Port_ID')['realizedProfit'].cumsum()
# Function to calculate metrics for each account
def calculate_metrics(df):
    metrics = {}

    # Total Realized Profit (PnL)
    total_realized_profit = df['realizedProfit'].sum()

    # Total Investment (sum of 'quantity' for BUY trades)
    total_investment = df[df['side'] == 'BUY']['quantity'].sum()

    # ROI
    roi = (total_realized_profit / total_investment) * 100 if total_investment != 0 else np.nan

    # Daily Returns
    df = df.sort_values(by='time')
    df['cumulative_pnl'] = df['realizedProfit'].cumsum()
    df['daily_return'] = df['cumulative_pnl'].pct_change().fillna(0)

    # Sharpe Ratio
    mean_daily_return = df['daily_return'].mean()
    std_daily_return = df['daily_return'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(90) if std_daily_return != 0 else np.nan

    # Maximum Drawdown (MDD)
    cumulative_pnl = df['cumulative_pnl']
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    # Win Rate and Number of Winning Positions
    winning_positions = df[df['realizedProfit'] > 0].shape[0]
    total_positions = df.shape[0]
    win_rate = (winning_positions / total_positions) * 100 if total_positions != 0 else np.nan

    # Store metrics
    metrics['Total Realized Profit'] = total_realized_profit
    metrics['ROI (%)'] = roi
    metrics['Sharpe Ratio'] = sharpe_ratio
    metrics['Maximum Drawdown'] = max_drawdown
    metrics['Win Rate (%)'] = win_rate
    metrics['Winning Positions'] = winning_positions
    metrics['Total Positions'] = total_positions

    return pd.Series(metrics)

# Group by Port_ID and compute metrics
account_metrics = trade_details.groupby('Port_ID', group_keys=False, include_groups=False).apply(calculate_metrics)

# Display the computed metrics
print(account_metrics.head())



# Example: Use ROI, Sharpe Ratio, and Total Realized Profit for ranking
ranking_metrics = account_metrics[['ROI (%)', 'Sharpe Ratio', 'Total Realized Profit']].copy()

# Normalize the metrics using min-max scaling
normalized = (ranking_metrics - ranking_metrics.min()) / (ranking_metrics.max() - ranking_metrics.min())

# Assign weights to each metric (adjust weights as needed)
weights = {
    'ROI (%)': 0.4,
    'Sharpe Ratio': 0.4,
    'Total Realized Profit': 0.2
}

# Compute a composite score
normalized['Composite Score'] = (normalized['ROI (%)'] * weights['ROI (%)'] +
                                 normalized['Sharpe Ratio'] * weights['Sharpe Ratio'] +
                                 normalized['Total Realized Profit'] * weights['Total Realized Profit'])

# Add the composite score back to the account_metrics DataFrame
account_metrics['Composite Score'] = normalized['Composite Score']

# Rank the accounts (higher composite score = better ranking)
account_metrics['Rank'] = account_metrics['Composite Score'].rank(ascending=False, method='min')

# Sort the DataFrame by rank
account_metrics.sort_values('Rank', inplace=True)

print(account_metrics.head())

top_20_accounts = account_metrics.nsmallest(20, 'Rank')
print("Top 20 Accounts:")
print(top_20_accounts)

# Save the account metrics to a CSV file
account_metrics.to_csv("account_metrics.csv", index=True)

# Save the top 20 ranked accounts to a CSV file
top_20_accounts.to_csv("top_20_accounts.csv", index=True)

