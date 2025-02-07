import yfinance as yf
import pandas as pd
from math import sqrt
from datetime import date
from typing import List


def get_schg_holdings() -> List:
    schg_holdings = pd.read_csv("downloaded_csv_files/SCHG_holdings.csv", skipfooter=8, engine='python')
    return schg_holdings['Symbol'].tolist()

def get_strong_buys(index="SCHG"):
    if index == "SP500":
        holdings = pd.read_csv("downloaded_csv_files/CSPX_holdings.csv", skiprows=2, engine='python')
        tickers = holdings['Ticker'].tolist()
    elif index == "SCHG":
        tickers = get_schg_holdings()
    else:
        # TODO: Throw exception
        pass
    stock_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            if stock.info is None:
                continue
            num_analysts = stock.info.get("numberOfAnalystOpinions", 0)
            recommendation = stock.info.get('recommendationKey', None)
            if num_analysts < 10 or recommendation != "strong_buy":
                continue
            market_cap = stock.info.get('marketCap', None)
            if market_cap < 10000000000:  # 10B
                continue
            name = stock.info.get('shortName', "noname")
            targets_mean = stock.analyst_price_targets['mean']
            # targets_median = stock.analyst_price_targets['median']
            current_price = stock.analyst_price_targets['current']
            targets_low = stock.analyst_price_targets['low']
            targets_high = stock.analyst_price_targets['high']
            upside = ((targets_mean - current_price) / current_price) * 100
            downside = (current_price - targets_low) / current_price * 100

            stock_data.append({
                "ticker": ticker,
                "name": name,
                "market_cap": round(market_cap / 1000000000, 2),
                "current_price": round(current_price, 2),
                "low": round(targets_low),
                "mean": round(targets_mean),
                "high": round(targets_high),
                "upside": round(upside, 2),
                "downside": round(downside, 2),
                "num_analysts": num_analysts,
                "recommendation": recommendation,
            })
        except:
            print(f"Exception for ticker: {ticker}. Continuing.")
    stock_info = pd.DataFrame(stock_data)
    stock_info.sort_values(by="market_cap", ascending=False, inplace=True)
    stock_info.reset_index(drop=True, inplace=True)
    print()
    print("Strong buys")
    print(stock_info)
    print()


def get_current_portfolio(stock_info: pd.DataFrame):
    INPUT_DIR = "input/"
    csv = INPUT_DIR+"stocks.csv"
    stocks_df = pd.read_csv(csv)
    current_stocks = stocks_df['Ticker'].tolist()
    current_portfolio = stock_info[stock_info['ticker'].isin(current_stocks)]
    current_portfolio.reset_index(drop=True, inplace=True)
    print("Current portfolio:")
    print(current_portfolio)


def adjust_weight(row):
    # penalty for high downside (-1/3)
    if row['downside'] > row['upside'] + 5:
        row['weight'] = row['weight'] / 1.5
    # premium for negative downside (+20%)
    if row['downside'] < 0:
        row['weight'] = row['weight'] * 1.2
    # premium for "strong buy" recommendation
    if row['recommendation'] == 'strong_buy':
        row['weight'] = row['weight'] * 1.1
    return row

def get_stock_info(tickers: List) -> pd.DataFrame:
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            name = stock.info['shortName']
        except:
            import pdb; pdb.set_trace()
        num_analysts = stock.info.get("numberOfAnalystOpinions", 0)
        if num_analysts == 0:
            print(f"Warning: no analysts for {name}")
            continue
        if num_analysts < 10:
            print(f"Skipping {name} - too few analysts")
            continue
        market_cap = stock.info.get('marketCap', None)
        if market_cap < 10000000000:  # 10B
            continue
        targets_mean = stock.analyst_price_targets['mean']
        # targets_median = stock.analyst_price_targets['median']
        current_price = stock.analyst_price_targets['current']
        targets_low = stock.analyst_price_targets['low']
        targets_high = stock.analyst_price_targets['high']
        upside = ((targets_mean - current_price) / current_price) * 100
        downside = (current_price - targets_low) / current_price * 100
        recommendation = stock.info.get('recommendationKey', None)

        stock_data.append({
            "ticker": ticker,
            "name": name[:15],
            "market_cap": round(market_cap / 1000000000, 2),
            "price": round(current_price, 2),
            "low": round(targets_low),
            # "mean": round(targets_mean),
            "high": round(targets_high),
            "upside": round(upside, 2),
            "downside": round(downside, 2),
            "analysts": num_analysts,
            "recommendation": recommendation,
        })
    stock_info = pd.DataFrame(stock_data)
    stock_info.sort_values(by="market_cap", ascending=False, inplace=True)
    stock_info.reset_index(drop=True, inplace=True)

    return stock_info

def get_stocks_with_high_upside(stock_info) -> pd.DataFrame:
    stocks_with_high_upside = stock_info[
          ((stock_info['upside'] > 10) & (stock_info['upside'] + 5 > stock_info['downside'])) \
        | (stock_info['upside'] > 15) \
        | (stock_info['downside'] < 5)]
    # Add 50B to make smaller positions more significant
    stocks_with_high_upside["weight"] = stocks_with_high_upside["market_cap"] + 100.0
    # Take sqrt to make smaller positions more significant
    stocks_with_high_upside["weight"] = stocks_with_high_upside["weight"] ** 0.4
    stocks_with_high_upside["weight"] = stocks_with_high_upside["weight"] * (stocks_with_high_upside["upside"] ** 0.5)

    stocks_with_high_upside = stocks_with_high_upside.apply(adjust_weight, axis=1)


    weight_sum = stocks_with_high_upside["weight"].sum()
    stocks_with_high_upside["weight"] = (stocks_with_high_upside["weight"] / weight_sum) * 100

    stocks_with_high_upside.sort_values(by="weight", ascending=False, inplace=True)
    stocks_with_high_upside.reset_index(drop=True, inplace=True)

    return stocks_with_high_upside

def main():
    TICKERS = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "AVGO", "ORCL", "V", "MA", "NFLX", "COST",
        "ACN", "MELI", "INTU", "AMD", "UBER", "CRM", "SPGI", "NU", "TTD",
        "CMG", "TXRH", "SNPS", "PANW",  "ISRG", "CRWD",  "BKNG", "HUBS",
        "AXON", "TEAM", "ASML", "ARM", "AMAT", "ADBE", "UNH", "ANET", "DDOG",
        "TSM", "SPOT", "NOW",  "NKE", "FICO", "LLY", "DASH", "IDXX",
        "MSCI", "MCO", "BLK", "VEEV", "WDAY", "SAP", "TOST", "PLTR",
        "JPM", "WFC", "BAC", "SCHW", "FI", "COF", "AXP", "GS", "SYK", "BSX", "APP", "ADP", "APH", "MRVL", "DELL", "VRT"
    ]
    # TICKERS = get_schg_holdings()
    # TICKERS += ['ASML', 'TSM', 'SAP', 'MELI', 'NU', 'ARM']
    # TICKERS = ['NVDA', 'MSFT', 'UNH', 'AMD', "UBER", 'ASML', 'ADBE', 'INTU', 'BLK', 'MELI', 'SPGI', 'LLY', 'SNPS', 'VEEV', 'WDAY', 'CRM', 'NU', 'CMG', 'FICO', 'IDXX']

    stock_info = get_stock_info(TICKERS)
    stocks_with_high_upside = get_stocks_with_high_upside(stock_info)

    pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns
    print()
    print("All stocks:")
    print(stock_info)
    print()
    print("Model portfolio:")
    print(stocks_with_high_upside)
    print()

    get_current_portfolio(stock_info)
    # get_strong_buys()


if __name__ == "__main__":
    main()
