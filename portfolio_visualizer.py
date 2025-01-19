import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

import numpy as np

import requests

urls = {
    'CSPX.L': "https://www.ishares.com/uk/individual/en/products/253743/ishares-sp-500-b-ucits-etf-acc-fund/1506575576011.ajax?fileType=csv&fileName=CSPX_holdings&dataType=fund",
    'IUIT.L': "https://www.ishares.com/uk/individual/en/products/280510/ishares-sp-500-information-technology-sector-ucits-etf/1506575576011.ajax?fileType=csv&fileName=IUIT_holdings&dataType=fund"
}

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

weights = {}
input_dir = "input/"
def handle_individual_stocks():
    csv = input_dir+"stocks.csv"
    stocks_df = pd.read_csv(csv)
    for _, row in stocks_df.iterrows():
        # Create a Ticker object
        ticker = yf.Ticker(row['Ticker'])

        # Get current stock price from the `info` method
        current_price = ticker.info['currentPrice']
        print(f"{row['Ticker']}: {current_price}")
        weights[row['Ticker']] = row['Shares']*current_price
        # print(f"{row['Ticker']}: {weights[row['Ticker']]}")

def handle_etfs():
    csv = input_dir+"etfs.csv"
    etfs_df = pd.read_csv(csv)
    for _, row in etfs_df.iterrows():
        # Create a Ticker object
        ticker = yf.Ticker(row['Ticker'])
        # Get current stock price from the `info` method
        current_price = ticker.info['previousClose']
        print(f"{row['Ticker']}: {current_price}")
        weights[row['Ticker']] = row['Shares']*current_price
        # print(f"{row['Ticker']}: {weights[row['Ticker']]}")

handle_individual_stocks()
handle_etfs()

weights_sum = sum(weights.values())

print(f"weights sum: {weights_sum}")
weights = {key: value / weights_sum for key, value in weights.items()}
print(weights)


for ticker in supported_tickers:
    save_file(urls[ticker], csv_files[ticker])

# Read the CSV file into a DataFrame
cspx_df = pd.read_csv(csv_files['CSPX.L'], skiprows=2)
iuit_df = pd.read_csv(csv_files['IUIT.L'], skiprows=2)

cspx_df = cspx_df[['Ticker', 'Weight (%)']]
raw_cspx_df = cspx_df.copy()
iuit_df = iuit_df[['Ticker', 'Weight (%)']]

# # Display the first few rows
# print(cspx_df.head())
# print(iuit_df.head())

cpsx_weight = weights['CSPX.L']
iuit_weight = weights['IUIT.L']

for df in [cspx_df, iuit_df]:
  df["SP500 weight"] = cspx_df["Weight (%)"]

# Normalize the weights so that the final weight adds up to 100%
cspx_df['Weight (%)'] = cspx_df['Weight (%)'] * cpsx_weight / (cpsx_weight+iuit_weight)
iuit_df['Weight (%)'] = iuit_df['Weight (%)'] * iuit_weight / (cpsx_weight+iuit_weight)

merged_df = pd.merge(cspx_df, iuit_df, on='Ticker', how='outer', suffixes=('_df1', '_df2'))
merged_df['Weight (%)'] = merged_df['Weight (%)_df1'].fillna(0) + merged_df['Weight (%)_df2'].fillna(0)
# merged_df = merged_df[['Ticker', 'Weight (%)']]

merged_df["SP500 weight"] = merged_df["SP500 weight_df1"]
merged_df.drop(columns=["SP500 weight_df1", "SP500 weight_df2"], inplace=True)
merged_df = merged_df.sort_values(by="Weight (%)", ascending=False)
merged_df.drop(columns=["Weight (%)_df1", "Weight (%)_df2"], inplace=True)

print(f"Sum of all weights is {merged_df['Weight (%)'].sum()}")
# print(merged_df.head(30))

df = merged_df.head(30)

# Set up the bar positions
y = np.arange(len(df)) * 100

plt.figure(figsize=(17, 12))

# Plot the horizontal bars
bars1 = plt.barh(y - 20, df['Weight (%)'], height=40, label='Weight (%)')
bars2 = plt.barh(y + 20, df['SP500 weight'], height=40, label='SP500 weight')

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
