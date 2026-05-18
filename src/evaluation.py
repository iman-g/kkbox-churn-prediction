# src/evaluation.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report)


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


def print_threshold_report(y_true: pd.Series,
                            oof_preds: np.ndarray,
                            threshold: float) -> None:
    """Print classification report at the tuned threshold."""
    y_pred = (oof_preds >= threshold).astype(int)
    auc    = roc_auc_score(y_true, oof_preds)
    ap     = average_precision_score(y_true, oof_preds)

    print(f"\nThreshold: {threshold:.3f}")
    print(f"OOF AUC: {auc:.5f} | Average Precision: {ap:.5f}")
    print(f"Lift over random: {ap / y_true.mean():.1f}x\n")
    print(classification_report(y_true, y_pred,
                                 target_names=['No Churn', 'Churn']))


def _make_serialisable(v):
    """Recursively convert to JSON-safe types."""
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {kk: _make_serialisable(vv) for kk, vv in v.items()}
    if isinstance(v, list):
        return [_make_serialisable(i) for i in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return None


def save_results(fold_results: list,
                 threshold: float,
                 oof_auc: float,
                 oof_ap: float,
                 output_path: str = 'results/cv_results.json') -> None:
    """Persist CV metrics to JSON for reproducibility."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    skip_keys = {'xgb_model', 'lgb_model', 'feature_names'}

    serialisable = {
        'oof_auc':   float(oof_auc),
        'oof_ap':    float(oof_ap),
        'threshold': float(threshold),
        'folds': []
    }

    for r in fold_results:
        fold_dict = {}
        for k, v in r.items():
            if k in skip_keys:
                continue
            fold_dict[k] = _make_serialisable(v)
        serialisable['folds'].append(fold_dict)

    with open(output_path, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"Results saved to {output_path}")
