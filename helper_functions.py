import requests
import yfinance as yf


def get_usd_to_eur_rate():
    # Ticker for USD to EUR exchange rate on Yahoo Finance
    ticker = "EUR=X"
    data = yf.Ticker(ticker)
    price = data.history(period="1d")["Close"]
    if not price.empty:
        return price.iloc[-1]  # Get the latest closing price
    else:
        print("Error: No data available.")
        return None

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