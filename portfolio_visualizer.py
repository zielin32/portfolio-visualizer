import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

import numpy as np

import requests

# Fetch the data
usd_eur_data = yf.Ticker("USDEUR=X")
# Get the current price
usd_eur = usd_eur_data.history(period="1d")["Close"].iloc[-1]
print(usd_eur)

urls = {
    'CSPX.L': "https://www.ishares.com/uk/individual/en/products/253743/ishares-sp-500-b-ucits-etf-acc-fund/1506575576011.ajax?fileType=csv&fileName=CSPX_holdings&dataType=fund",
    'IUIT.L': "https://www.ishares.com/uk/individual/en/products/280510/ishares-sp-500-information-technology-sector-ucits-etf/1506575576011.ajax?fileType=csv&fileName=IUIT_holdings&dataType=fund",
    'CNDX.L': "https://www.ishares.com/uk/individual/en/products/253741/ishares-nasdaq-100-ucits-etf/1506575576011.ajax?fileType=csv&fileName=CNDX_holdings&dataType=fund"
}
# aliases
urls["SP500"] = urls["CSPX.L"]
urls["IT"] = urls["IUIT.L"]
urls["Nasdaq100"] = urls["CNDX.L"]
urls["XNAS.DE"] = urls["CNDX.L"]

supported_tickers = urls.keys()

csv_dir = "downloaded_csv_files/"
csv_files = {}
for ticker in supported_tickers:
    csv_files[ticker] = csv_dir+f"{ticker}_holdings.csv"


def save_file(url, file_name):
  try:
      # Send GET request
      response = requests.get(url)
      response.raise_for_status()  # Raise an exception for HTTP errors
      
      # Write content to file
      with open(file_name, "wb") as file:
          file.write(response.content)
      
      print(f"File downloaded successfully and saved as '{file_name}'.")
  except requests.exceptions.RequestException as e:
      print(f"An error occurred: {e}")

input_dir = "input/"
stock_weights = {}
def handle_individual_stocks():
    csv = input_dir+"stocks.csv"
    stocks_df = pd.read_csv(csv)
    for _, row in stocks_df.iterrows():
        # Create a Ticker object
        ticker = yf.Ticker(row['Ticker'])

        # Get current stock price from the `info` method
        current_price = ticker.info['currentPrice']
        print(f"{row['Ticker']}: {current_price}")
        stock_weights[row['Ticker']] = row['Shares']*current_price

etf_weights = {}
def handle_etfs():
    csv = input_dir+"etfs.csv"
    etfs_df = pd.read_csv(csv)
    for _, row in etfs_df.iterrows():
        # Create a Ticker object
        ticker = yf.Ticker(row['Ticker'])
        # Get current stock price from the `info` method
        current_price = ticker.info['previousClose']
        if row['Currency'] == "EUR":
            current_price *= usd_eur
        print(f"{row['Ticker']}: {current_price}")
        etf_weights[row['Ticker']] = row['Shares']*current_price
    etfs_df.drop(columns=["Currency"], inplace=True)

handle_individual_stocks()
handle_etfs()

weights_sum = sum(stock_weights.values()) + sum(etf_weights.values())
stock_weights = {key: (value / weights_sum) for key, value in stock_weights.items()}
etf_weights = {key: value / weights_sum for key, value in etf_weights.items()}

weights = stock_weights | etf_weights

print(f"weights sum: {weights_sum}")
print(weights)

# TODO: dotąd wszystko ok z wagami, ale dalej coś się psuje

for ticker in supported_tickers:
    save_file(urls[ticker], csv_files[ticker])

# Read the CSV files into DataFrames
dataframes = {}
for ticker in ['CSPX.L', 'IUIT.L', 'XNAS.DE']:
    dataframes[ticker] = pd.read_csv(csv_files[ticker], skiprows=2)

cspx_df = dataframes['CSPX.L'][['Ticker', 'Weight (%)']]
iuit_df = dataframes['IUIT.L'][['Ticker', 'Weight (%)']]
xnas_df = dataframes['XNAS.DE'][['Ticker', 'Weight (%)']]

data = {
    "Ticker": stock_weights.keys(),
    "Weight (%)": [value * 100 for value in stock_weights.values()]
}
stock_weights_df = pd.DataFrame(data)

for df in [cspx_df, iuit_df, xnas_df, stock_weights_df]:
  df.loc[:, "SP500 Weight (%)"] = cspx_df["Weight (%)"]

# ######## TODO: need to rewrite this
# # Normalize the weights so that the final weight adds up to 100%
# cspx_df['Weight (%)'] = cspx_df['Weight (%)'] * cpsx_weight
# iuit_df['Weight (%)'] = iuit_df['Weight (%)'] * iuit_weight

# merged_df = pd.merge(cspx_df, iuit_df, stock_weights_df, on='Ticker', how='outer', suffixes=('_df1', '_df2', '_df3'))
# merged_df['Weight (%)'] = merged_df['Weight (%)_df1'].fillna(0) + merged_df['Weight (%)_df2'].fillna(0) + merged_df['Weight (%)_df3'].fillna(0)
# # merged_df = merged_df[['Ticker', 'Weight (%)']]

# merged_df["SP500 weight"] = merged_df["SP500 weight_df1"]
# merged_df.drop(columns=["SP500 weight_df1", "SP500 weight_df2", "SP500 weight_df3"], inplace=True)
# merged_df = merged_df.sort_values(by="Weight (%)", ascending=False)
# merged_df.drop(columns=["Weight (%)_df1", "Weight (%)_df2", "Weight (%)_df3"], inplace=True)
# ########## End of TODO

def merge_and_normalize(etf_dfs, stock_dfs, merge_column='Ticker', weight_column='Weight (%)_df'):
    for etf in etf_dfs.items():
        etf[1]['Weight (%)'] *= etf_weights[etf[0]]
    dfs = list(etf_dfs.values()) + [stock_dfs]

    # Ensure that the first DataFrame is the starting point
    merged_df = dfs[0]
    
    # Sequentially merge the remaining DataFrames
    for i, df in enumerate(dfs[1:]):
        merged_df = pd.merge(merged_df, df, on=merge_column, how='outer', suffixes=(f'_df1', f'_df2'))
        # Sum the weight columns, filling NaN with 0
        weight_columns = [col for col in merged_df.columns if weight_column in col and "SP500" not in col]
        columns_to_drop = [col for col in merged_df.columns if weight_column in col]
        merged_df['Weight (%)'] = merged_df[weight_columns].fillna(0).sum(axis=1)
        merged_df['SP500 Weight (%)'] = merged_df['SP500 Weight (%)_df1']
        # Drop the intermediate weight columns
        merged_df.drop(columns=columns_to_drop, inplace=True)
    
    # Optionally, sort the result by 'Weight (%)'
    merged_df = merged_df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)
    
    # Drop any other unnecessary columns, for example, 'SP500 weight_df1', etc.
    suffixes_to_drop = [col for col in merged_df.columns if col.endswith('df1') or col.endswith('df2') or col.endswith('df3')]
    merged_df.drop(columns=suffixes_to_drop, inplace=True)
    
    return merged_df

# List of DataFrames to merge
etf_dfs = {'CSPX.L': cspx_df, 'IUIT.L': iuit_df, 'XNAS.DE': xnas_df}

# Call the function
merged_df = merge_and_normalize(etf_dfs, stock_weights_df)

checksum = merged_df['Weight (%)'].sum()
assert checksum > 99.8 and checksum < 100.2, "Sum of all weights does not add up to 100"
print(f"Sum of all weights is {checksum}")

df = merged_df.head(30)

# Set up the bar positions
y = np.arange(len(df)) * 100

plt.figure(figsize=(17, 12))

# Plot the horizontal bars
bars1 = plt.barh(y - 20, df['Weight (%)'], height=40, label='Weight (%)')
bars2 = plt.barh(y + 20, df['SP500 Weight (%)'], height=40, label='SP500 Weight (%)')

# Add numbers next to the bars
for bar in bars1:
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}', va='center', ha='left', fontsize=10)

for bar in bars2:
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}', va='center', ha='left', fontsize=10)

# Customize the plot
plt.xlabel('Weight')
plt.ylabel('Ticker')
plt.title('Weight vs SP500 Weight for Tickers')
plt.yticks(y, df['Ticker'])
plt.legend()

# Flip the y-axis upside down
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()
