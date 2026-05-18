
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

from src.config import RESULTS, RANDOM_STATE
from src.data import load_all
from src.features import build_feature_matrix
from src.survival import (compute_subscription_gaps, fit_kaplan_meier,
                           derive_churn_threshold, segment_by_frequency,
                           compute_segment_thresholds,
                           plot_survival_overview, plot_segment_thresholds)
from src.model import (train_cv, tune_threshold, print_cv_summary,
                        plot_model_performance, plot_feature_importance)


def main(rapid: bool = False):

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    labels, members, tx = load_all()

    print(tx['transaction_date'].dt.date.value_counts().sort_index().tail(20))
    print(tx['transaction_date'].dt.date.value_counts().sort_index().head(20))

    if rapid:
        sample_users = labels['msno'].sample(frac=0.1, random_state=RANDOM_STATE)
        labels  = labels[labels['msno'].isin(sample_users)]
        members = members[members['msno'].isin(sample_users)]
        tx      = tx[tx['msno'].isin(sample_users)]
        print(f"RAPID MODE: {len(labels):,} users")

    # ── 2. Survival analysis ──────────────────────────────────────────
    print("\n" + "="*50)
    print("PHASE 1: SURVIVAL ANALYSIS")
    print("="*50)

    gaps       = compute_subscription_gaps(tx)
    kmf        = fit_kaplan_meier(gaps)
    thresholds = derive_churn_threshold(kmf)

    gaps       = segment_by_frequency(gaps, labels)
    seg_summary = compute_segment_thresholds(gaps)

    plot_survival_overview(kmf, thresholds, segment_gaps=gaps)
    plot_segment_thresholds(seg_summary)

    # ── 3. Feature engineering ────────────────────────────────────────
    print("\n" + "="*50)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*50)

    X, y, feature_cols = build_feature_matrix(labels, members, tx)

    # ── 4. Model training ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("PHASE 3: MODEL TRAINING")
    print("="*50)

    fold_results, oof_preds = train_cv(X, y)
    print_cv_summary(fold_results)

    threshold = tune_threshold(y, oof_preds, target_recall=0.6)

    # ── 5. Evaluation plots ───────────────────────────────────────────
    plot_model_performance(y, oof_preds, fold_results, threshold)
    plot_feature_importance(fold_results)

    # ── 6. Save artifacts for dashboard ──────────────────────────────
    artifacts = {
        'fold_results':   fold_results,
        'feature_cols':   feature_cols,
        'threshold':      threshold,
        'kmf':            kmf,
        'thresholds':     thresholds,
        'seg_summary':    seg_summary,
        'oof_preds':      oof_preds,
        'y_true':         y,
    }
    with open(RESULTS / 'artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\nArtifacts saved to results/artifacts.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rapid', action='store_true',
                        help='Use 10%% of users for fast smoke-test')
    args = parser.parse_args()
    main(rapid=args.rapid)
