import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from typing import Dict, List, Tuple

import numpy as np
from helper_functions import *

benchmark_str = "SP500"
benchmark_weight_str = benchmark_str + " Weight (%)"

URLS = {
    'CSPX.L': "https://www.ishares.com/uk/individual/en/products/253743/ishares-sp-500-b-ucits-etf-acc-fund/1506575576011.ajax?fileType=csv&fileName=CSPX_holdings&dataType=fund",
    'IUIT.L': "https://www.ishares.com/uk/individual/en/products/280510/ishares-sp-500-information-technology-sector-ucits-etf/1506575576011.ajax?fileType=csv&fileName=IUIT_holdings&dataType=fund",
    'CNDX.L': "https://www.ishares.com/uk/individual/en/products/253741/ishares-nasdaq-100-ucits-etf/1506575576011.ajax?fileType=csv&fileName=CNDX_holdings&dataType=fund",
    'QDVG.DE': "https://www.ishares.com/uk/individual/en/products/280507/ishares-sp-500-health-care-sector-ucits-etf/1506575576011.ajax?fileType=csv&fileName=IUHC_holdings&dataType=fund"
}

# This is because it's easier to download holdings data for iShares ETFs
# and both ETFs follow the same index
URLS["XNAS.DE"] = URLS["CNDX.L"]
URLS["WTEF.DE"] = URLS["CSPX.L"]
URLS["IUHC.L"] = URLS["QDVG.DE"]

supported_etfs = ("CSPX.L", "IUIT.L", "XNAS.DE", "WTEF.DE", "QDVG.DE", "IUHC.L")
INPUT_DIR = "input/"

aliases = {"SP500": "CSPX.L", "Nasdaq100": "XNAS.DE", "IT": "IUIT.L"}

def _normalize_stock_weights_in_etfs(etf_dfs: Dict[str, pd.DataFrame],
                                     etf_weights: Dict[str, np.float64]
                                     ) -> None:
    """
    Multiply the weight of each stock in each ETF by the weight of that ETF in the whole portfolio.
    For example, if a weight of an S&P500 ETF in a portfolio is 45%,
    the weight of each stock in that ETF is going to be multiplied by 0.45
    """
    for etf in etf_dfs.items():
        etf[1]['Weight (%)'] *= etf_weights[etf[0]]

def merge_and_normalize(etf_dfs: Dict[str, pd.DataFrame],
                        stock_dfs: pd.DataFrame,
                        etf_weights: Dict[str, np.float64],
                        merge_column='Ticker',
                        weight_column='Weight (%)_df'
                        ) -> pd.DataFrame:
   
    # Individual stock weights don't need any normalization
    # because each stock already has the proper weight in the portfolio
    _normalize_stock_weights_in_etfs(etf_dfs, etf_weights)

    # Here, benchmark_etf if popped from the dictionary of supported ETFs,
    # and will be used as a base for merging all dataframes toghether.
    # It is important that this base is the benchmark, because outherwise
    # benchmark weights will not be assigned correctly.
    benchmark_etf = etf_dfs.pop(aliases[benchmark_str])
    combined_dfs_list = list(etf_dfs.values()) + [stock_dfs]

    # Ensure that the first DataFrame is the starting point
    merged_df = benchmark_etf
    
    # Sequentially merge the remaining DataFrames
    for i, df in enumerate(combined_dfs_list):
        merged_df = pd.merge(merged_df, df, on=merge_column, how='outer', suffixes=(f'_df1', f'_df2'))
        # Sum the weight columns, filling NaN with 0
        weight_columns = [col for col in merged_df.columns if weight_column in col and benchmark_str not in col]
        columns_to_drop = [col for col in merged_df.columns if weight_column in col]
        merged_df['Weight (%)'] = merged_df[weight_columns].fillna(0).sum(axis=1)
        merged_df[benchmark_weight_str] = merged_df[benchmark_weight_str+'_df1']
        # Drop the intermediate weight columns
        merged_df.drop(columns=columns_to_drop, inplace=True)
    
    # Sort the result by 'Weight (%)'
    merged_df = merged_df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)
    
    # Drop any other unnecessary columns, for example, 'SP500 weight_df1', etc.
    suffixes_to_drop = [col for col in merged_df.columns if col.endswith('df1') or col.endswith('df2') or col.endswith('df3')]
    merged_df.drop(columns=suffixes_to_drop, inplace=True)
    
    return merged_df

def download_holdings_data() -> Dict[str, str]:
    """Download ETF holdings data and save to CSV files."""
    CSV_DIR = "downloaded_csv_files/"
    csv_files = {}
    for ticker in URLS.keys():
        csv_files[ticker] = f"{CSV_DIR}{ticker}_holdings.csv"
        save_file(URLS[ticker], csv_files[ticker])
    return csv_files

def get_stock_weights():
    stock_weights = {}
    csv = INPUT_DIR+"stocks.csv"
    stocks_df = pd.read_csv(csv)
    for _, row in stocks_df.iterrows():
        # Create a Ticker object
        ticker = yf.Ticker(row['Ticker'])

        # Get current stock price from the `info` method
        current_price = ticker.info['currentPrice']
        print(f"{row['Ticker']}: {current_price}")
        stock_weights[row['Ticker']] = float(row['Shares'])*current_price
    return stock_weights

def get_etf_weights():
    etf_weights = {}
    usd_eur = get_usd_to_eur_rate()
    csv = INPUT_DIR+"etfs.csv"
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
        if row['Ticker'] == "WTEF.DE":
            etf_weights[row['Ticker']] *= 0.9  # Efficient Core ETFs have 90% stock allocation
    etfs_df.drop(columns=["Currency"], inplace=True)
    return etf_weights

def normalize_weights(stock_weights, etf_weights):
    weights_sum = sum(stock_weights.values()) + sum(etf_weights.values())
    print(f"weights sum: {weights_sum}")

    stock_weights = {key: (value / weights_sum) for key, value in stock_weights.items()}
    etf_weights = {key: value / weights_sum for key, value in etf_weights.items()}
    return stock_weights, etf_weights

def get_normalized_weights():
    stock_weights = get_stock_weights()
    etf_weights = get_etf_weights()

    return normalize_weights(stock_weights, etf_weights)

def read_etfs(csv_files):
    # Read the CSV files into DataFrames
    dataframes = {}
    for ticker in supported_etfs:
        dataframes[ticker] = pd.read_csv(csv_files[ticker], skiprows=2)

    etf_dfs = {}
    for ticker in supported_etfs:
        etf_dfs[ticker] = dataframes[ticker][['Ticker', 'Weight (%)']]

    return etf_dfs

def plot(df):
    # Set up the bar positions
    y = np.arange(len(df)) * 100

    plt.figure(figsize=(17, 12))

    # Plot the horizontal bars
    bars1 = plt.barh(y - 20, df['Weight (%)'], height=40, label='Weight (%)')
    bars2 = plt.barh(y + 20, df[benchmark_weight_str], height=40, label=benchmark_weight_str)

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
    plt.title('Weight vs ' + benchmark_weight_str + ' for Tickers')
    plt.yticks(y, df['Ticker'])
    plt.legend()

    # Flip the y-axis upside down
    plt.gca().invert_yaxis()

    # Show the plot
    plt.tight_layout()
    plt.show()

def add_benchmark_weight_to_all_dataframes(etf_dfs: Dict[str, pd.DataFrame], stock_weights_df: pd.DataFrame) -> None:
    print(f"type etf_dfs: {type(etf_dfs)}")
    print(f"type stock_weights_df: {type(stock_weights_df)}")
    etf_and_stocks_dataframes_list = list(etf_dfs.values()) + [stock_weights_df]
    for df in etf_and_stocks_dataframes_list:
        df.loc[:, benchmark_weight_str] = etf_dfs[aliases[benchmark_str]]["Weight (%)"]

def main():
    csv_files = download_holdings_data()

    stock_weights, etf_weights = get_normalized_weights()

    etf_dfs = read_etfs(csv_files)

    stock_weights_data = {
        "Ticker": stock_weights.keys(),
        "Weight (%)": [value * 100 for value in stock_weights.values()]
    }
    stock_weights_df = pd.DataFrame(stock_weights_data)

    add_benchmark_weight_to_all_dataframes(etf_dfs, stock_weights_df)

    # Call the function
    merged_df = merge_and_normalize(etf_dfs, stock_weights_df, etf_weights)

    checksum = merged_df['Weight (%)'].sum()
    print(f"Sum of all weights is {checksum}")
    assert checksum > 99.8 and checksum < 100.2, "Sum of all weights does not add up to 100"

    plot(merged_df.head(30))

if __name__ == "__main__":
    main()
