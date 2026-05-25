import pandas as pd

SCORED_PATH = 'data/news_scored_2021_2023.parquet'
PRICES_PATH = 'data/features_prices_2021_2023.parquet'
OUTPUT_PATH = 'data/feature_table_2021_2023.parquet'

def main():
    scored = pd.read_parquet(SCORED_PATH)
    print(f"  {len(scored):,} scored articles loaded")

    before = len(scored)
    scored = scored.drop_duplicates(
        subset=['stock_symbol', 'trading_day', 'article_title']
    )
    print(f"  Dropped {before - len(scored):,} duplicate articles "
          f"(now {len(scored):,})")

    daily = (
        scored
        .groupby(['stock_symbol', 'trading_day'], as_index=False)
        .agg(
            sent_mean_1d=('sent_score', 'mean'),
            sent_std_1d=('sent_score', 'std'),
            sent_count_1d=('sent_score', 'size'),
        )
    )
    daily['sent_std_1d'] = daily['sent_std_1d'].fillna(0)
    print(f"  {len(daily):,} (ticker, day) cells with articles")

    prices = pd.read_parquet(PRICES_PATH)
    full_grid = prices[['ticker', 'date']].rename(
        columns={'ticker': 'stock_symbol', 'date': 'trading_day'}
    )

    daily['trading_day'] = (
        pd.to_datetime(daily['trading_day']).astype('datetime64[ns]')
    )
    full_grid['trading_day'] = (
        pd.to_datetime(full_grid['trading_day']).astype('datetime64[ns]')
    )

    daily = full_grid.merge(
        daily,
        on=['stock_symbol', 'trading_day'],
        how='left',
    )

    daily['sent_count_1d'] = daily['sent_count_1d'].fillna(0).astype(int)
    daily['sent_mean_1d']  = daily['sent_mean_1d'].fillna(0)
    daily['sent_std_1d']   = daily['sent_std_1d'].fillna(0)

    n_with_news = (daily['sent_count_1d'] > 0).sum()
    print(f"{len(daily)} cells "
          f"({n_with_news/len(daily):.1%} have articles)")

    daily = daily.sort_values(['stock_symbol', 'trading_day']).reset_index(drop=True)

    daily['sent_sum_1d'] = daily['sent_mean_1d'] * daily['sent_count_1d']

    daily['_sum_5d'] = (
        daily.groupby('stock_symbol')['sent_sum_1d']
             .transform(lambda x: x.rolling(5, min_periods=1).sum())
    )
    daily['sent_count_5d'] = (
        daily.groupby('stock_symbol')['sent_count_1d']
             .transform(lambda x: x.rolling(5, min_periods=1).sum())
    )

    daily['sent_mean_5d'] = (
        daily['_sum_5d'] / daily['sent_count_5d']
    ).fillna(0)

    daily['sent_momentum_5d'] = (
        daily.groupby('stock_symbol')['sent_mean_5d']
             .transform(lambda x: x - x.shift(5))
             .fillna(0)
    )

    daily = daily.drop(columns=['sent_sum_1d', '_sum_5d'])
    daily = daily.rename(columns={'stock_symbol': 'ticker', 'trading_day': 'date'})

    prices['date'] = pd.to_datetime(prices['date']).astype('datetime64[ns]')
    daily['date']  = pd.to_datetime(daily['date']).astype('datetime64[ns]')

    df = prices.merge(daily, on=['ticker', 'date'], how='left')

    print(f"final feature table shape {df.shape}")
    print(f"columns {df.columns.tolist()}")
    print(f"nan per column")
    print(df.isna().sum())

    sent_cols = ['sent_mean_1d', 'sent_std_1d', 'sent_count_1d',
                 'sent_mean_5d', 'sent_count_5d', 'sent_momentum_5d']
    print(f"sentiment feature distribution")
    print(df[sent_cols].describe())

    print(f"target balance {df['target_direction'].mean():.3%} up days")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()