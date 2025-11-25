# portfolio return calculator
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
import traceback
import os

try:
    with open('practice/data/stocks.json', 'r') as f:
        data_file = json.load(f)
except FileNotFoundError:
    data_file = {}

stock_data_list = {}

for i in data_file:
    try:
        df = yf.download(i, start="2020-01-01", end="2024-01-01", auto_adjust=True)
        if df is None or df.empty or 'Close' not in df.columns:
            print(f"No data returned for {i}, skipping.")
            continue
        stock_data_list[i] = df
        print(f"Fetched data for {i}: ", stock_data_list[i].head())
    except Exception:
        print(f"Error fetching data for {i}:")
        traceback.print_exc()

def plot_all_stocks(stock_data):
    if not stock_data:
        print("No stock data available to plot.")
        return

    series_list = []
    for ticker, df in stock_data.items():
        try:
            if 'Close' in df.columns:
                s = df['Close']
            elif isinstance(df.columns, pd.MultiIndex):
                extracted = None
                for lvl in (0, 1):
                    try:
                        tmp = df.xs('Close', axis=1, level=lvl, drop_level=True)
                        extracted = tmp
                        break
                    except Exception:
                        continue

                if extracted is None:
                    matches = [col for col in df.columns if 'Close' in col]
                    if matches:
                        extracted = df[matches[0]]

                if extracted is None:
                    print(f"Warning: couldn't find Close for {ticker}, skipping")
                    continue

                s = extracted
            else:
                print(f"Warning: unexpected columns for {ticker}, skipping")
                continue

            if isinstance(s, pd.DataFrame):
                if ticker in s.columns:
                    s = s[ticker]
                elif s.shape[1] == 1:
                    s = s.iloc[:, 0]
                else:\
                    s = s.iloc[:, 0]

            s.name = ticker
            series_list.append(s)
        except Exception:
            print(f"Error extracting Close for {ticker}:")
            traceback.print_exc()

    if not series_list:
        print("No Close series extracted from stock data.")
        return

    prices = pd.concat(series_list, axis=1)

    prices = prices.dropna(axis=1, how='all')
    if prices.empty:
        print("No valid price series to plot after cleaning.")
        return

    normalized = prices.div(prices.iloc[0]).mul(100)

    ax = normalized.plot(figsize=(12, 6))
    ax.set_title('Stock prices normalized to 100 (start = 100)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (base 100)')
    ax.grid(True)
    ax.legend(title='Ticker')
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        try:
            out_path = os.path.join('practice', 'normalized_prices.png')
            fig = ax.get_figure()
            fig.savefig(out_path, bbox_inches='tight')
            print(f"Plot saved to {out_path}")
        except Exception:
            print("Failed to display or save the plot:")
            traceback.print_exc()


def pr(prices, weights):
    if prices is None or prices.empty:
        raise ValueError("`prices` must be a non-empty DataFrame")

    returns = prices.pct_change().dropna()

    if isinstance(weights, dict):
        w = pd.Series(weights)
        w = w.reindex(prices.columns).fillna(0)
    else:
        w = pd.Series(weights, index=prices.columns)

    if w.sum() != 0:
        w = w / w.sum()

    portfolio_daily = returns.dot(w)
    cumulative = (1 + portfolio_daily).cumprod() - 1

    return portfolio_daily, cumulative