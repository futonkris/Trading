import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
from scipy.stats import chi2

FEATURE_TABLE = 'data/feature_table_2021_2023.parquet'
RANDOM_SEED   = 42

TRAIN_END = '2022-12-31'
VAL_END   = '2023-06-30'

BASE_FEATURES = [
    'ticker',  # categorical
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'vol_5', 'vol_20',
    'rsi_14',
    'volume_zscore_20',
    'ret_spy_1', 'ret_spy_5',
]

SENTIMENT_FEATURES = [
    'sent_mean_1d', 'sent_std_1d', 'sent_count_1d',
    'sent_mean_5d', 'sent_count_5d', 'sent_momentum_5d',
]

TARGET = 'target_direction'

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': RANDOM_SEED,
    'enable_categorical': True,
    'early_stopping_rounds': 30,
}

def make_splits(df):
    train = df[df['date'] <= TRAIN_END].copy()
    val   = df[(df['date'] > TRAIN_END) & (df['date'] <= VAL_END)].copy()
    test  = df[df['date'] > VAL_END].copy()

    return train, val, test

def prep_features(df, feature_cols):
    X = df[feature_cols].copy()
    if 'ticker' in X.columns:
        X['ticker'] = X['ticker'].astype('category')
    y = df[TARGET].astype(int)

    return X, y

def train_one(name, train, val, test, feature_cols):
    print(f"features ({len(feature_cols)}), {feature_cols}")

    X_train, y_train = prep_features(train, feature_cols)
    X_val,   y_val   = prep_features(val,   feature_cols)
    X_test,  y_test  = prep_features(test,  feature_cols)

    print(f"train shape - {X_train.shape}, val shape - {X_val.shape}, test shape - {X_test.shape}")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"stopped at iteration {model.best_iteration} of {XGB_PARAMS['n_estimators']}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"test accuracy {acc:.4f}")
    print(f"test AUC {auc:.4f}")
    print(f"confusion matrix (rows=true, cols=pred) \n{confusion_matrix(y_test, y_pred)}")
    print(f"classification report \n{classification_report(y_test, y_pred, digits=4)}")

    return {
        'name': name,
        'model': model,
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': acc,
        'auc': auc,
    }

def mcnemar(baseline_correct, augmented_correct):
    b = ((baseline_correct == True)  & (augmented_correct == False)).sum()
    c = ((baseline_correct == False) & (augmented_correct == True)).sum()

    if (b + c) == 0:
        return {'b': 0, 'c': 0, 'statistic': 0.0, 'p_value': 1.0}
    
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(statistic, df=1)

    return {'b': int(b), 'c': int(c),
            'statistic': float(statistic), 'p_value': float(p_value)}

def plot_roc(baseline, augmented, path):
    fpr_b, tpr_b, _ = roc_curve(baseline['y_test'],  baseline['y_pred_proba'])
    fpr_a, tpr_a, _ = roc_curve(augmented['y_test'], augmented['y_pred_proba'])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC = {baseline['auc']:.3f})",  lw=2)
    plt.plot(fpr_a, tpr_a, label=f"Augmented (AUC = {augmented['auc']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: price-only vs price + sentiment')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved ROC plot to {path}")

def main():
    print("feature table")
    df = pd.read_parquet(FEATURE_TABLE)
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    train, val, test = make_splits(df)
    print(f"train - {len(train):>5,} ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"val - {len(val):>5,} ({val['date'].min().date()} → {val['date'].max().date()})")
    print(f"test - {len(test):>5,} ({test['date'].min().date()} → {test['date'].max().date()})")

    base_rate = test[TARGET].mean()
    print(f"test base rate {base_rate:.4f}") # any meaningful model needs to beat this

    baseline  = train_one('baseline (price only)',
                          train, val, test, BASE_FEATURES)
    augmented = train_one('aug (price + sentiment)',
                          train, val, test, BASE_FEATURES + SENTIMENT_FEATURES)

    print(f"{'Model':<32} {'Accuracy':<12} {'AUC':<12}")
    print(f"{'Always-up baseline':<32} {base_rate:<12.4f} {'0.5000':<12}")
    print(f"{'Baseline (price only)':<32} {baseline['accuracy']:<12.4f} {baseline['auc']:<12.4f}")
    print(f"{'Augmented (+ sentiment)':<32} {augmented['accuracy']:<12.4f} {augmented['auc']:<12.4f}")
    delta_acc = augmented['accuracy'] - baseline['accuracy']
    delta_auc = augmented['auc'] - baseline['auc']
    print(f"{'delta from sentiment':<32} {delta_acc:+.4f} {delta_auc:+.4f}")

    baseline_correct  = (baseline['y_pred']  == baseline['y_test']).values
    augmented_correct = (augmented['y_pred'] == augmented['y_test']).values
    mc = mcnemar(baseline_correct, augmented_correct)

    print(f"Baseline right, Augmented wrong: {mc['b']}")
    print(f"Baseline wrong, Augmented right: {mc['c']}")
    print(f"Statistic: {mc['statistic']:.4f}    p-value: {mc['p_value']:.4f}")
    if mc['p_value'] < 0.05:
        print(f"  → Statistically significant difference (p < 0.05)")
    else:
        print(f"  → Not statistically significant (p ≥ 0.05)")

    booster = augmented['model'].get_booster()
    imp = booster.get_score(importance_type='gain')
    imp_df = (
        pd.DataFrame(imp.items(), columns=['feature', 'importance'])
          .sort_values('importance', ascending=False)
          .reset_index(drop=True)
    )
    print(f"top 15 feature importance")
    print(imp_df.head(15).to_string(index=False))
    imp_df.to_csv('data/feature_importance.csv', index=False)

    sentiment_in_top10 = imp_df.head(10)['feature'].isin(SENTIMENT_FEATURES).sum()
    print(f"sentiment features in top 10 {sentiment_in_top10}/{len(SENTIMENT_FEATURES)}")

    plot_roc(baseline, augmented, 'data/roc_comparison.png')

    for name, result in [('baseline', baseline), ('augmented', augmented)]:
        preds = test[['date', 'ticker', 'target_direction']].copy()
        preds['pred_proba'] = result['y_pred_proba']
        preds['pred_label'] = result['y_pred']
        preds.to_parquet(f'data/predictions_{name}.parquet', index=False)

    results = {
        'splits': {
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'test_base_rate': float(base_rate),
        },
        'baseline': {
            'accuracy': float(baseline['accuracy']),
            'auc': float(baseline['auc']),
            'best_iteration': int(baseline['model'].best_iteration),
        },
        'augmented': {
            'accuracy': float(augmented['accuracy']),
            'auc': float(augmented['auc']),
            'best_iteration': int(augmented['model'].best_iteration),
        },
        'delta': {
            'accuracy': float(delta_acc),
            'auc': float(delta_auc),
        },
        'mcnemar': mc,
        'sentiment_features_in_top10': int(sentiment_in_top10),
    }
    with open('data/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nartifacts saved -")
    print(f"data/results.json")
    print(f"data/feature_importance.csv")
    print(f"data/roc_comparison.png")
    print(f"data/predictions_baseline.parquet")
    print(f"data/predictions_augmented.parquet")

if __name__ == '__main__':
    main()