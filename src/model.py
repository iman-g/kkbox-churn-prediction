
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, roc_curve,
                              precision_recall_curve)
import xgboost as xgb
import lightgbm as lgb

from src.config import XGB_PARAMS, LGB_PARAMS, CV_FOLDS, COLORS, RANDOM_STATE


# ── Cross-validation ──────────────────────────────────────────────────────────

def train_cv(X: pd.DataFrame, y: pd.Series) -> tuple[list, np.ndarray]:

    skf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}\nFOLD {fold}/{CV_FOLDS}\n{'='*50}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # XGBoost
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
        xgb_preds = xgb_model.predict(X_val)
        xgb_auc   = roc_auc_score(y_val, xgb_preds)
        print(f"  XGBoost AUC: {xgb_auc:.5f}")

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        lgb_auc   = roc_auc_score(y_val, lgb_preds)
        print(f"  LightGBM AUC: {lgb_auc:.5f}")

        # Inverse-AUC ensemble (higher AUC = more weight)
        w_xgb   = xgb_auc
        w_lgb   = lgb_auc
        w_total = w_xgb + w_lgb

        ensemble_preds = (w_xgb * xgb_preds + w_lgb * lgb_preds) / w_total
        ensemble_auc   = roc_auc_score(y_val, ensemble_preds)
        print(f"  Ensemble AUC: {ensemble_auc:.5f}")

        oof_preds[val_idx] = ensemble_preds

        fold_results.append({
            'fold':          fold,
            'xgb_model':     xgb_model,
            'lgb_model':     lgb_model,
            'xgb_auc':       xgb_auc,
            'lgb_auc':       lgb_auc,
            'ensemble_auc':  ensemble_auc,
            'xgb_weight':    w_xgb / w_total,
            'lgb_weight':    w_lgb / w_total,
            'feature_names': list(X.columns),
        })

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ap  = average_precision_score(y, oof_preds)
    print(f"\nOOF AUC: {oof_auc:.5f}")
    print(f"OOF Average Precision: {oof_ap:.5f}")

    return fold_results, oof_preds


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(y_true: pd.Series,
                    oof_preds: np.ndarray,
                    target_recall: float = 0.6) -> float:

    precision, recall, thresholds = precision_recall_curve(y_true, oof_preds)

    valid = recall[:-1] >= target_recall
    if not valid.any():
        print(f"Warning: target recall {target_recall} not achievable. "
              f"Using max recall threshold.")
        return float(thresholds[0])

    best_threshold = thresholds[valid][np.argmax(precision[:-1][valid])]
    best_precision = precision[:-1][valid][np.argmax(precision[:-1][valid])]
    best_recall    = recall[:-1][valid][np.argmax(precision[:-1][valid])]

    print(f"\nOptimal threshold: {best_threshold:.3f}")
    print(f"  Precision: {best_precision:.3f} | Recall: {best_recall:.3f}")

    return float(best_threshold)


# ── Evaluation ────────────────────────────────────────────────────────────────

def print_cv_summary(fold_results: list) -> None:
    xgb_aucs = [r['xgb_auc'] for r in fold_results]
    lgb_aucs = [r['lgb_auc'] for r in fold_results]
    ens_aucs = [r['ensemble_auc'] for r in fold_results]

    print("\n" + "=" * 55)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 55)
    print(f"{'Model':<15} {'Mean AUC':<12} {'Std':<10} {'Best':<10}")
    print("-" * 50)
    for name, aucs in [('XGBoost', xgb_aucs),
                        ('LightGBM', lgb_aucs),
                        ('Ensemble', ens_aucs)]:
        print(f"{name:<15} {np.mean(aucs):<12.5f} "
              f"{np.std(aucs):<10.5f} {max(aucs):<10.5f}")
    print("=" * 55)


def plot_model_performance(y_true: pd.Series,
                            oof_preds: np.ndarray,
                            fold_results: list,
                            threshold: float) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Churn Prediction Model Performance', fontsize=15,
                 fontweight='bold')

    # ── ROC curve ────────────────────────────────────────────────────
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, oof_preds)
    auc = roc_auc_score(y_true, oof_preds)
    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2,
            label=f'OOF AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Out-of-Fold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Precision-Recall curve ────────────────────────────────────────
    ax = axes[0, 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, oof_preds)
    ap = average_precision_score(y_true, oof_preds)
    ax.plot(recall, precision, color=COLORS['secondary'], linewidth=2,
            label=f'AP = {ap:.4f}')
    ax.axhline(y_true.mean(), linestyle='--', color='gray', alpha=0.5,
               label=f'Baseline (churn rate = {y_true.mean():.1%})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (Out-of-Fold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── AUC per fold ──────────────────────────────────────────────────
    ax = axes[1, 0]
    folds    = [r['fold'] for r in fold_results]
    xgb_aucs = [r['xgb_auc'] for r in fold_results]
    lgb_aucs = [r['lgb_auc'] for r in fold_results]
    ens_aucs = [r['ensemble_auc'] for r in fold_results]

    x = np.arange(len(folds))
    w = 0.25
    ax.bar(x - w, xgb_aucs, w, label='XGBoost', color=COLORS['primary'])
    ax.bar(x,     lgb_aucs, w, label='LightGBM', color=COLORS['secondary'])
    ax.bar(x + w, ens_aucs, w, label='Ensemble', color=COLORS['accent'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('AUC')
    ax.set_title('AUC by Model and Fold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(min(xgb_aucs) - 0.01, max(ens_aucs) + 0.01)

    # ── Score distribution ────────────────────────────────────────────
    ax = axes[1, 1]
    churners     = oof_preds[y_true == 1]
    non_churners = oof_preds[y_true == 0]
    ax.hist(non_churners, bins=50, alpha=0.6, color=COLORS['primary'],
            label='Non-churners', density=True)
    ax.hist(churners,     bins=50, alpha=0.6, color=COLORS['danger'],
            label='Churners', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.3f}')
    ax.set_xlabel('Predicted Churn Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/model_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/model_performance.png")


def plot_feature_importance(fold_results: list, top_n: int = 20) -> None:

    feature_names = fold_results[0]['feature_names']


    lgb_importances = np.mean([
        r['lgb_model'].feature_importances_ for r in fold_results
    ], axis=0)

    importance_df = pd.DataFrame({
        'feature':    feature_names,
        'importance': lgb_importances,
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['importance'],
            color=COLORS['primary'], alpha=0.8)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (LightGBM gain)')
    ax.set_title(f'Top {top_n} Features — Averaged Across {len(fold_results)} Folds')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/feature_importance.png")