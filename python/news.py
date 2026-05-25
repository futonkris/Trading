import pandas as pd

NEWS_PATH    = 'data/news_fnspid_2021_2023.parquet'
PRICES_PATH  = 'data/features_prices_2021_2023.parquet'
OUTPUT_PATH  = 'data/news_aligned_2021_2023.parquet'

def main():
    news = pd.read_parquet(NEWS_PATH)
    print(f"  News shape: {news.shape}")
    print(f"  News columns: {news.columns.tolist()}")

    print("load prices for trading calendar")
    prices = pd.read_parquet(PRICES_PATH)
    trading_days = (
        prices['date']
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    print(f"trading calendar {len(trading_days)} days"
          f"({trading_days.min().date()} → {trading_days.max().date()})")

    news['date'] = pd.to_datetime(news['date']).astype('datetime64[ns]')
    news = news.rename(columns={'date': 'news_date'})
    news = news.sort_values('news_date').reset_index(drop=True)

    print("aligned news and dates")
    trading_df = pd.DataFrame({
        'trading_day': pd.to_datetime(trading_days).astype('datetime64[ns]')
    })
    trading_df = trading_df.sort_values('trading_day').reset_index(drop=True)

    news_aligned = pd.merge_asof(
        news,
        trading_df,
        left_on='news_date',
        right_on='trading_day',
        direction='forward',
        allow_exact_matches=True,
    )

    before = len(news_aligned)
    news_aligned = news_aligned.dropna(subset=['trading_day'])
    print(f"dropped {before - len(news_aligned)} articles")

    print(f"final article level shape {news_aligned.shape}")
    print(f"articles per ticker \n{news_aligned['stock_symbol'].value_counts()}")

    daily_counts = (
        news_aligned.groupby(['stock_symbol', 'trading_day'])
                    .size()
                    .reset_index(name='article_count')
    )
    print(f"articles per ticker and trading day")
    print(daily_counts['article_count'].describe())

    n_tickers = news_aligned['stock_symbol'].nunique()
    n_days = news_aligned['trading_day'].nunique()
    n_cells = n_tickers * n_days
    coverage = len(daily_counts) / n_cells if n_cells else 0
    print(f"coverage {len(daily_counts)}/{n_cells} cells have ≥1 article "
          f"({coverage:.1%})")

    news_aligned.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved {OUTPUT_PATH}")

if __name__ == '__main__':
    main()