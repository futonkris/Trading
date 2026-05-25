import sys
from pathlib import Path

import features as stage_features       
import news as stage_news           
import sentiment as stage_sentiment     
import build as stage_feature_table 
import models as stage_models       
import backtest as stage_backtest       
import shapAnalysis as stage_shap            


STAGES = [
    ('1. Build price features',
     'data/features_prices_2021_2023.parquet',
     stage_features.main),
    ('2. Align news to trading days',
     'data/news_aligned_2021_2023.parquet',
     stage_news.main),
    ('3. Score sentiment (FinBERT)',
     'data/news_scored_2021_2023.parquet',
     stage_sentiment.main),
    ('4. Build feature table',
     'data/feature_table_2021_2023.parquet',
     stage_feature_table.main),
    ('5. Train models',
     'data/predictions_augmented.parquet',
     stage_models.main),
    ('6. Backtest strategies',
     'data/backtest_results.csv',
     stage_backtest.main),
    ('7. SHAP analysis',
     'data/shap_summary.png',
     stage_shap.main),
]

def main():
    force = '--force' in sys.argv

    fnspid_path = Path('data/news_fnspid_2021_2023.parquet')
    if not fnspid_path.exists():
        print(f"missing fnspid")
        sys.exit(1)

    for label, output, fn in STAGES:
        print(f"{label}")

        if not force and Path(output).exists():
            print(f"skipped, output exists")
            print(f"use --force to rerun all stages")
            continue

        try:
            fn()
        except Exception as e:
            print(f"stage failed {type(e).__name__}: {e}")
            sys.exit(1)

    print("\nresults in data")
    print("results.json - headline metrics")
    print("feature_importance.csv - gain-based importance")
    print("shap_importance.csv - SHAP-based importance")
    print("backtest_results.csv - strategy comparison")
    print("roc_comparison.png - model ROC curves")
    print("backtest_curves.png - cumulative-return curves")
    print("shap_summary.png - SHAP beeswarm")

if __name__ == '__main__':
    main()