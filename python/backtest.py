import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

PRED_BASELINE   = 'data/predictions_baseline.parquet'
PRED_AUGMENTED  = 'data/predictions_augmented.parquet'
FEATURES_PATH   = 'data/features_prices_2021_2023.parquet'
OUTPUT_CSV      = 'data/backtest_results.csv'
OUTPUT_PNG      = 'data/backtest_curves.png'

LONG_THRESHOLD  = 0.5
TOP_K           = 3
TRADING_DAYS_YR = 252
RISK_FREE_ANN   = 0.04   

def attach_target_returns(preds, features):
    feats = features[['date', 'ticker', 'target_return']].copy()
    feats['date'] = pd.to_datetime(feats['date']).astype('datetime64[ns]')

    preds['date'] = pd.to_datetime(preds['date']).astype('datetime64[ns]')

    merged = preds.merge(feats, on=['date', 'ticker'], how='left')
    merged['target_simple_return'] = np.exp(merged['target_return']) - 1

    return merged

def long_only_strategy(preds, threshold=0.5, top_k=None):
    daily = []

    for date, group in preds.groupby('date'):
        if top_k is not None:
            selected = group.nlargest(top_k, 'pred_proba')
        else:
            selected = group[group['pred_proba'] > threshold]
        daily_ret = selected['target_simple_return'].mean() if len(selected) else 0.0
        daily.append({'date': date, 'return': daily_ret, 'n_positions': len(selected)})
        
    return pd.DataFrame(daily).sort_values('date').reset_index(drop=True)

def basket_strategy(preds):
    daily = (
        preds
        .groupby('date')['target_simple_return']
        .mean()
        .reset_index()
        .rename(columns={'target_simple_return': 'return'})
    )

    daily['n_positions'] = 8

    return daily.sort_values('date').reset_index(drop=True)

def spy_benchmark(test_dates):
    start = test_dates.min()
    end   = test_dates.max() + pd.Timedelta(days=10)
    spy = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy_df = pd.DataFrame({
        'date':  pd.to_datetime(spy.index).astype('datetime64[ns]'),
        'close': spy['Close'].values,
    })
    
    spy_df['simple'] = spy_df['close'].pct_change()
    spy_df['return'] = spy_df['simple'].shift(-1)

    spy_df = spy_df[spy_df['date'].isin(test_dates)][['date', 'return']].copy()
    spy_df['n_positions'] = 1

    return spy_df.sort_values('date').reset_index(drop=True)

def compute_metrics(returns_df, name):
    r = returns_df['return'].fillna(0).values
    cum = np.cumprod(1 + r)
    total_ret = cum[-1] - 1

    years = len(r) / TRADING_DAYS_YR
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    vol_ann = np.std(r) * np.sqrt(TRADING_DAYS_YR)

    excess = r - (RISK_FREE_ANN / TRADING_DAYS_YR)
    sharpe = (np.mean(excess) / np.std(excess) * np.sqrt(TRADING_DAYS_YR)
              if np.std(excess) > 0 else 0)

    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    nonzero = (r != 0).sum()
    win_rate = (r > 0).sum() / nonzero if nonzero > 0 else 0

    return {
        'strategy': name,
        'total_return': total_ret,
        'cagr': cagr,
        'volatility_annual': vol_ann,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_positions_per_day': returns_df['n_positions'].mean(),
        'days_in_market_pct': (returns_df['n_positions'] > 0).mean(),
        '_cumulative_curve': cum,
        '_dates': returns_df['date'].values,
    }

def main():
    preds_base = pd.read_parquet(PRED_BASELINE)
    preds_aug = pd.read_parquet(PRED_AUGMENTED)
    features = pd.read_parquet(FEATURES_PATH)

    preds_base = attach_target_returns(preds_base, features)
    preds_aug = attach_target_returns(preds_aug, features)

    test_dates = pd.Series(sorted(preds_base['date'].unique()))
    print(f"test period {test_dates.min().date()} → {test_dates.max().date()} "
          f"({len(test_dates)} trading days)")

    strategies = {
        'SPY': spy_benchmark(test_dates),
        'EqualWeight': basket_strategy(preds_base),
        'Baseline': long_only_strategy(preds_base, threshold=LONG_THRESHOLD),
        'Augmented': long_only_strategy(preds_aug,  threshold=LONG_THRESHOLD),
        'Augmented_Top3': long_only_strategy(preds_aug,  top_k=TOP_K),
    }

    all_metrics = [compute_metrics(df, name) for name, df in strategies.items()]

    print(f"backtest results h2 2023")
    print(f"{'Strategy':<18} {'Total':>9} {'CAGR':>9} {'Vol':>9} "
          f"{'Sharpe':>8} {'MaxDD':>9} {'Win%':>8} {'%InMkt':>8}")
    for m in all_metrics:
        print(
            f"{m['strategy']:<18} "
            f"{m['total_return']:>9.2%} "
            f"{m['cagr']:>9.2%} "
            f"{m['volatility_annual']:>9.2%} "
            f"{m['sharpe']:>8.2f} "
            f"{m['max_drawdown']:>9.2%} "
            f"{m['win_rate']:>8.1%} "
            f"{m['days_in_market_pct']:>8.1%}"
        )

    out_df = pd.DataFrame([
        {k: v for k, v in m.items() if not k.startswith('_')}
        for m in all_metrics
    ])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved metrics to {OUTPUT_CSV}")

    plt.figure(figsize=(11, 6.5))
    for m in all_metrics:
        dates  = np.concatenate([[m['_dates'][0]], m['_dates']])
        curve  = np.concatenate([[1.0], m['_cumulative_curve']])
        plt.plot(dates, curve, lw=2,
                 label=f"{m['strategy']} ({m['total_return']:+.1%}, Sharpe={m['sharpe']:.2f})")
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Cumulative return (starting capital = 1)')
    plt.title('Backtest: cumulative returns, H2 2023')
    plt.legend(loc='best', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"saved diagrams to {OUTPUT_PNG}")

if __name__ == '__main__':
    main()