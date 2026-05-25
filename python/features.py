import os
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ['NVDA', 'AMZN', 'TSLA', 'GOOG', 'AAPL', 'AMD', 'INTC', 'MSFT']
START = '2020-12-01'
END   = '2024-01-15'   
OUTPUT_PATH = 'data/features_prices_2021_2023.parquet'

def rsi(close, length=14):
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def zscore_20(s):
    return (s - s.rolling(20).mean()) / s.rolling(20).std()

def main():
    os.makedirs('data', exist_ok=True)

    data = yf.download(
        TICKERS + ['SPY'],
        start=START,
        end=END,
        auto_adjust=True,    
        progress=False,
    )
    print(f"raw shape - {data.shape}")

    df = data.stack(level='Ticker', future_stack=True).reset_index()
    df.columns = [c.lower() for c in df.columns]
    if 'level_0' in df.columns:
        df = df.rename(columns={'level_0': 'date'})
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    print(f"long format {df.shape}")

    df['log_close'] = np.log(df['close'])

    df['ret_1'] = df.groupby('ticker')['log_close'].diff(1)
    for n in [5, 10, 20]:
        df[f'ret_{n}'] = df.groupby('ticker')['log_close'].diff(n)

    for n in [5, 20]:
        df[f'vol_{n}'] = (
            df.groupby('ticker')['ret_1']
              .transform(lambda x: x.rolling(n).std())
        )

    df['rsi_14'] = (
        df.groupby('ticker')['close']
          .transform(lambda x: rsi(x, length=14))
    )

    df['volume_zscore_20'] = (
        df.groupby('ticker')['volume']
          .transform(zscore_20)
    )

    df['target_return'] = df.groupby('ticker')['ret_1'].shift(-1)
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    df.loc[df['target_return'].isna(), 'target_direction'] = np.nan

    print("spy")
    spy = df[df['ticker'] == 'SPY'][['date', 'ret_1', 'ret_5']].copy()
    spy = spy.rename(columns={'ret_1': 'ret_spy_1', 'ret_5': 'ret_spy_5'})
    df = df[df['ticker'] != 'SPY'].copy()
    df = df.merge(spy, on='date', how='left')

    df = df[
        (df['date'] >= '2021-01-01') &
        (df['date'] <= '2023-12-31')
    ].copy()

    feature_cols = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'vol_5', 'vol_20', 'rsi_14', 'volume_zscore_20',
        'ret_spy_1', 'ret_spy_5',
    ]
    nan_before = len(df)
    df = df.dropna(subset=feature_cols + ['target_direction'])
    print(f"dropped {nan_before - len(df)} rows with NaN in features/target")

    print(f"final shape {df.shape}")
    print(f"Per-ticker counts:\n{df.groupby('ticker').size()}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Target balance (% up days): {df['target_direction'].mean():.3%}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()