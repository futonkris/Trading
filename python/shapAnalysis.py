import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

FEATURE_TABLE = 'data/feature_table_2021_2023.parquet'
RANDOM_SEED   = 42

TRAIN_END = '2022-12-31'
VAL_END   = '2023-06-30'

ALL_FEATURES = [
    'ticker',
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'vol_5', 'vol_20',
    'rsi_14',
    'volume_zscore_20',
    'ret_spy_1', 'ret_spy_5',
    'sent_mean_1d', 'sent_std_1d', 'sent_count_1d',
    'sent_mean_5d', 'sent_count_5d', 'sent_momentum_5d',
]

SENTIMENT_FEATURES = [
    'sent_mean_1d', 'sent_std_1d', 'sent_count_1d',
    'sent_mean_5d', 'sent_count_5d', 'sent_momentum_5d',
]

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

def prep(d):
    X = d[ALL_FEATURES].copy()
    X['ticker'] = X['ticker'].astype('category')
    y = d['target_direction'].astype(int)

    return X, y


def main():
    print("feature table")
    df = pd.read_parquet(FEATURE_TABLE).dropna(subset=['target_direction'])
    print(f"{len(df)} rows")

    train = df[df['date'] <= TRAIN_END]
    val   = df[(df['date'] > TRAIN_END) & (df['date'] <= VAL_END)]
    test  = df[df['date'] > VAL_END]
    X_train, y_train = prep(train)
    X_val,   y_val   = prep(val)
    X_test,  y_test  = prep(test)

    print("training aug model")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"best itertaion {model.best_iteration}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"shap shape {shap_values.shape}")

    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=len(ALL_FEATURES), show=False)
    plt.tight_layout()
    plt.savefig('data/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("saved shap_summary.png")

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type='bar',
                      max_display=len(ALL_FEATURES), show=False)
    plt.tight_layout()
    plt.savefig('data/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("saved shap_bar.png")

    mean_abs = np.abs(shap_values).mean(axis=0)
    sent_idx = [i for i, f in enumerate(X_test.columns) if f in SENTIMENT_FEATURES]
    top_sent = sorted(sent_idx, key=lambda i: -mean_abs[i])[:2]

    for idx in top_sent:
        feat = X_test.columns[idx]
        plt.figure()
        try:
            shap.dependence_plot(feat, shap_values, X_test,
                                 interaction_index='auto', show=False)
        except Exception as e:
            plt.close()
            continue
        plt.tight_layout()
        plt.savefig(f'data/shap_dependence_{feat}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved shap_dependence_{feat}.png")

    imp = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_abs,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    imp.to_csv('data/shap_importance.csv', index=False)

    print(f"mean shap importance")
    print(imp.head(15).to_string(index=False))
    print(f"saved shap_importance.csv")


if __name__ == '__main__':
    main()