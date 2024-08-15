import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from itertools import product
import dash_bootstrap_components as dbc
from dash import html, dash_table
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from dash import html
from dash import dcc
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output, State, ALL
import requests
from bs4 import BeautifulSoup
from dash import html, dash_table
from dash_bootstrap_templates import load_figure_template
from datetime import datetime, timedelta
from itertools import product
import json
from io import StringIO

combined_data = None

def download_data_fillna(tickers, start_date, end_date):
    valid_tickers = []
    young_tickers = []
    invalid_tickers = []

    """
    Download historical stock data for a list of tickers, categorize tickers based on the data availability, and handle missing values.

    This function downloads closing price data for each ticker between the specified start and end dates using the `yfinance` library. It categorizes tickers into valid, young, or invalid based on the availability of data and the age of the data. Missing values in the data are forward-filled to ensure continuity.

    Args:
        tickers (list of str): A list of stock tickers to download data for.
        start_date (str): The start date for the data download in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data download in 'YYYY-MM-DD' format.

    Returns:
        tuple:
            - pd.DataFrame: A DataFrame containing the closing price data for valid tickers with forward-filled missing values.
            - list of str: A list of tickers categorized as young due to insufficient historical data.
            - list of str: A list of tickers categorized as invalid due to errors or lack of data.

    Used resources to download the data from Yahoo Finance:
    1. https://www.tutorialspoint.com/get-financial-data-from-yahoo-finance-with-python
    2. https://medium.com/@kasperjuunge/yfinance-10-ways-to-get-stock-data-with-python-6677f49e8282
    """

    for ticker in tickers:
        try:
            data = yf.download(tickers=[ticker], start=start_date, end=end_date)['Close']
            print(f"Downloaded data for {ticker}: {data}")

            if data.empty:
                print(f"{ticker} has no data, categorizing as invalid.")
                invalid_tickers.append(ticker)
                continue  

            if data.index[0] > pd.to_datetime("2023-01-03"):
                print(f"{ticker} is young, categorizing as young.")
                young_tickers.append(ticker)
            else:
                data = pd.DataFrame(data.rename(ticker))
                valid_tickers.append(data)
                print(f"{ticker} is valid and added to the list of valid tickers.")

        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            invalid_tickers.append(ticker)

    if valid_tickers:
        all_data = pd.concat(valid_tickers, axis=1)
        all_data = all_data.asfreq('B') 

        data = all_data.ffill()

        return data, young_tickers, invalid_tickers
    else:
        return pd.DataFrame(), young_tickers, invalid_tickers
    
def adf_test(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in the time series.

    Args:
        series (pd.Series): The time series data to test.

    Returns:
        float: The p-value of the ADF test. A p-value less than 0.05 indicates stationarity.
    """
    result = adfuller(series, autolag='AIC')
    return result[1]

def forecast_arima(data, steps=183):
    """
    Forecast future stock prices using the ARIMA model for each ticker in the dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the historical stock price data.
        steps (int): Number of periods to forecast into the future (default is 183 days).

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with the forecasted stock prices.
            - dict: A dictionary containing the ARIMA parameters (p, d, q) used for each ticker.

    GenAI: Function written by GenAI with minor adaptations
    """
    print("The forecast is being calculated. Please wait.")
    forecasted_data = pd.DataFrame()
    arima_params = {}

    for ticker in data.columns:
        series = data[ticker].dropna()

        if adf_test(series) > 0.05:
            d = 1
        else:
            d = 0

        best_aic = float("inf")
        best_order = (0, d, 0)
        model = None
        
        try:
            p, q = 0, 0
            while True:
                try:
                    temp_model = auto_arima(series, start_p=p, start_q=q, max_p=p+1, max_q=q+1, d=d, seasonal=False, 
                                            stepwise=False, suppress_warnings=True, error_action="ignore", trace=False)
                    temp_aic = temp_model.aic()
                    
                    if temp_aic < best_aic:
                        best_aic = temp_aic
                        best_order = temp_model.order
                        model = temp_model
                        p, q = p + 1, q + 1
                    else:
                        break
                except Exception as e:
                    print(f"Failed to fit ARIMA model for {ticker} with p={p}, q={q}: {e}")
                    break

            forecast = model.predict(n_periods=steps)
            forecasted_data[ticker] = forecast
            arima_params[ticker] = best_order
        except Exception as e:
            print(f"Failed to fit ARIMA model for {ticker}: {e}")

    return forecasted_data, arima_params
    
def calculate_daily_returns(data):
    returns = data.pct_change().dropna()
    return returns

def calculate_statistics(returns):
    mean_returns = returns.mean()
    corr_matrix = returns.corr()
    cov_matrix = returns.cov()
    return mean_returns, corr_matrix, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_stddev

# GenAI: Function written by GenAI with minor adaptations
def objective_function(weights, mean_returns, cov_matrix, risk_preference):
    portfolio_return, portfolio_stddev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(portfolio_return * risk_preference - portfolio_stddev * (1 - risk_preference))

def check_sum(weights):
    return np.sum(weights) - 1

# GenAI: Function written by GenAI with minor adaptations
def optimize_portfolio(mean_returns, cov_matrix, risk_preference):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_preference)

    constraints = {'type': 'eq', 'fun': check_sum}
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    result = minimize(objective_function, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# GenAI: Function written by GenAI with minor adaptations
def get_optimal_portfolio(tickers, start_date, risk_preference):
    data = download_data_fillna(tickers, start_date="2023-01-03", end_date=datetime.today()-timedelta(days=1))
    forecasted_data, arima_params = forecast_arima(data)
    combined_data = pd.concat([data, forecasted_data], axis=0)
    daily_returns = calculate_daily_returns(combined_data)
    mean_returns, corr_matrix, cov_matrix = calculate_statistics(daily_returns)

    optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix, risk_preference=risk_preference)


TOOLTIP_TEXT = {
    "Market Cap": "The total market value of a company's outstanding shares. Indicates the size of the company and is used to compare companies within the same industry.",
    "Trailing P/E": "Price-to-earnings ratio based on the last 12 months of actual earnings. Helps investors understand how much they are paying for a company's earnings. A high P/E might indicate high future growth expectations, while a low P/E might indicate the opposite.",
    "PEG Ratio": "Price/earnings-to-growth ratio, which factors in expected earnings growth. Helps determine if a stock is over or undervalued considering its earnings growth",
    "Price/Sales": "The ratio of a company's stock price to its revenues. Useful for evaluating companies that are not yet profitable. It shows how much investors are willing to pay per dollar of sales.",
    "Enterprise Value": "The total value of a company, including debt and excluding cash.",
    "EV/Revenue": "The ratio of enterprise value to revenue.  Indicates how much investors are willing to pay for each dollar of revenue, providing insight into a company's valuation relative to its sales."
}

def get_trending_tickers():
    """
    Fetches trending stock tickers from Yahoo Finance.

    Returns:
        list: A list of dictionaries containing ticker symbols of trending stocks.

    GenAI: Function written by GenAI with some adaptations
    """
    url = "https://finance.yahoo.com/trending-tickers/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad response status codes

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find('div', {'id': 'list-res-table'})  # Update class name here
        
        if table is None:
            raise ValueError("Unable to find table with class 'W100'")

        rows = table.find_all('tr')
        
        trending_tickers_data = []
        for row in rows[1:]:  # skipping header row
            columns = row.find_all("td")
            ticker = columns[0].text.strip()

            trending_tickers_data.append({
                "Ticker": ticker
            })

        return trending_tickers_data

    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def format_market_cap(value):
    """
    Formats the market capitalization value into a readable string.

    Args:
        value (str): The market cap value as a string.

    Returns:
        str: Formatted market cap with appropriate suffix (T, B, M) or "N/A" if not available.
    """
    if value == "N/A":
        return value

    value = float(value)
    if value >= 1e12:  # Trillions
        return f'{value / 1e12:.3f}T'
    elif value >= 1e11:  # Hundreds of Billions
        return f'{value / 1e9:.3f}B'
    elif value >= 1e10:  # Tens of Billions
        return f'{value / 1e9:.3f}B'
    elif value >= 1e9:  # Billions
        return f'{value / 1e9:.3f}B'
    elif value >= 1e6:  # Millions
        return f'{value / 1e6:.3f}M'
    else:
        return str(value)
    
def format_enterprise_value(value):
    """
    Formats the enterprise value into a readable string.

    Args:
        value (str): The enterprise value as a string.

    Returns:
        str: Formatted enterprise value with appropriate suffix (T, B, M) or "N/A" if not available.
    """
    if value == "N/A":
        return value
    
    value = float(value)
    if value >= 1e12:  # Trillions
        return f'{value / 1e12:.2f}T'
    elif value >= 1e11:  # Hundreds of Billions
        return f'{value / 1e9:.2f}B'
    elif value >= 1e10:  # Tens of Billions
        return f'{value / 1e9:.2f}B'
    elif value >= 1e9:  # Billions
        return f'{value / 1e9:.2f}B'
    elif value >= 1e6:  # Millions
        return f'{value / 1e6:.2f}M'
    else:
        return str(value)

def get_key_statistics(tickers):
    """
    Retrieves key financial statistics for a list of tickers.

    Args:
        tickers (list of str): List of stock tickers.

    Returns:
        dict: A dictionary where each key is a ticker symbol and each value is another dictionary containing key statistics.
    
    GenAI: Function written by GenAI with some adaptations
    """
    statistics = {}
    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)
        stats = ticker_data.info
        key_stats = {
            "Market Cap": format_market_cap(stats.get("marketCap", "N/A")),
            "Trailing P/E": round(stats.get("trailingPE", "N/A"), 2) if stats.get("trailingPE", "N/A") != "N/A" else "N/A",
            "PEG Ratio": round(stats.get("pegRatio", "N/A"), 2) if stats.get("pegRatio", "N/A") != "N/A" else "N/A",
            "Price/Sales": round(stats.get("priceToSalesTrailing12Months", "N/A"), 2) if stats.get("priceToSalesTrailing12Months", "N/A") != "N/A" else "N/A",
            "Enterprise Value": format_enterprise_value(stats.get("enterpriseValue", "N/A")),
            "EV/Revenue": round(stats.get("enterpriseToRevenue", "N/A"), 2) if stats.get("enterpriseToRevenue", "N/A") != "N/A" else "N/A"
        }
        statistics[ticker] = key_stats
    return statistics

def generate_ticker_rectangles():
    tickers = get_trending_tickers()[:24]  # Limit to the first 24 tickers
    max_ticker_length = max(len(ticker['Ticker']) for ticker in tickers)

    """
    Generates a list of styled HTML div elements representing the trending tickers.

    Returns:
        list: A list of HTML div elements, each containing a ticker symbol.
    """
    
    return [
        html.Div(
            ticker['Ticker'],
            style={
                'padding': '20px',
                'margin': '10px',
                'backgroundColor': 'rgba(47, 79, 79, 0.6)',  # Slate theme background color with 60% opacity
                'color': '#00ffff',
                'borderRadius': '10px',
                'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.3)',
                'textAlign': 'center',
                'width': f'{max_ticker_length * 15}px',  # Adjust width based on ticker length
                'fontFamily': 'Lato, monospace',
                'font-weight': 'bold',
                'display': 'flex',  # Use Flexbox
                'justifyContent': 'center',  # Center horizontally
                'alignItems': 'center'  # Center vertically
            }
        ) for ticker in tickers
    ]