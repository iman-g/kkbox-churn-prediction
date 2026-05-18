
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features import build_rfm_features, build_subscription_features
from src.survival import compute_subscription_gaps, fit_kaplan_meier
from src.config import FEATURE_CUTOFF


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_fake_transactions(n_users: int = 100,
                            n_tx_per_user: int = 5) -> pd.DataFrame:

    np.random.seed(42)
    rows = []
    base_date = pd.Timestamp('2016-01-01')

    for uid in range(n_users):
        user_id = f"USER_{uid:04d}"
        tx_date = base_date + timedelta(days=int(np.random.randint(0, 200)))

        for _ in range(n_tx_per_user):
            plan_days = int(np.random.choice([30, 90, 365]))
            list_price = np.random.choice([99, 149, 180, 298])
            rows.append({
                'msno':                   user_id,
                'payment_method_id':      np.random.randint(30, 45),
                'payment_plan_days':      plan_days,
                'plan_list_price':        list_price,
                'actual_amount_paid':     list_price * np.random.uniform(0.8, 1.0),
                'is_auto_renew':          np.random.randint(0, 2),
                'transaction_date':       tx_date,
                'membership_expire_date': tx_date + timedelta(days=plan_days),
                'is_cancel':              np.random.randint(0, 2),
            })
            tx_date += timedelta(days=plan_days + np.random.randint(-5, 10))

    return pd.DataFrame(rows)


def make_fake_members(n_users: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        'msno':                   [f"USER_{i:04d}" for i in range(n_users)],
        'city':                    np.random.randint(1, 22, n_users),
        'bd':                      np.random.randint(18, 60, n_users),
        'gender':                  np.random.choice(['male', 'female', None], n_users),
        'registered_via':          np.random.randint(3, 13, n_users),
        'registration_init_time':  [20150101 + i * 100 for i in range(n_users)],
    })


# ── Feature tests ─────────────────────────────────────────────────────────────

def test_rfm_features_shape():
    tx = make_fake_transactions(n_users=50)
    rfm = build_rfm_features(tx)
    assert len(rfm) == 50, "One RFM row per user"
    assert 'recency_days' in rfm.columns
    assert 'frequency' in rfm.columns
    assert 'total_paid' in rfm.columns


def test_rfm_no_nulls():
    tx = make_fake_transactions(n_users=50)
    rfm = build_rfm_features(tx)
    assert rfm.isnull().sum().sum() == 0, "RFM features must have no nulls"


def test_rfm_recency_non_negative():
    tx = make_fake_transactions(n_users=50)
    rfm = build_rfm_features(tx)
    assert (rfm['recency_days'] >= 0).all(), "Recency must be non-negative"


def test_subscription_features_shape():
    tx = make_fake_transactions(n_users=50)
    sub = build_subscription_features(tx)
    assert len(sub) == 50, "One subscription row per user"
    assert 'cancel_rate' in sub.columns
    assert 'discount_rate' in sub.columns
    assert 'tenure_days' in sub.columns


def test_discount_rate_bounded():
    tx = make_fake_transactions(n_users=100)
    sub = build_subscription_features(tx)
    assert (sub['discount_rate'] >= 0).all(), "Discount rate must be >= 0"
    assert (sub['discount_rate'] <= 1).all(), "Discount rate must be <= 1"


def test_cancel_rate_bounded():
    tx = make_fake_transactions(n_users=100)
    sub = build_subscription_features(tx)
    assert (sub['cancel_rate'] >= 0).all()
    assert (sub['cancel_rate'] <= 1).all()


def test_tenure_non_negative():
    tx = make_fake_transactions(n_users=100)
    sub = build_subscription_features(tx)
    assert (sub['tenure_days'] >= 0).all(), "Tenure must be non-negative"


# ── Feature cutoff test ───────────────────────────────────────────────────────

def test_feature_cutoff_enforced():
    """
    Features must only use transactions before FEATURE_CUTOFF.
    This is the critical leakage prevention test.
    """
    tx = make_fake_transactions(n_users=50)


    future_tx = tx.copy()
    future_tx['transaction_date'] = pd.Timestamp('2017-03-15')
    future_tx['actual_amount_paid'] = 99999  # obviously wrong value

    tx_combined = pd.concat([tx, future_tx], ignore_index=True)

    rfm_all    = build_rfm_features(tx_combined)
    rfm_before = build_rfm_features(tx)


    assert (rfm_all['total_paid'] == rfm_before['total_paid']).all(), \
        "Future transactions must be excluded by FEATURE_CUTOFF"


# ── Survival analysis tests ───────────────────────────────────────────────────

def test_subscription_gaps_columns():
    tx = make_fake_transactions(n_users=50)
    gaps = compute_subscription_gaps(tx)
    assert 'duration' in gaps.columns
    assert 'event' in gaps.columns
    assert (gaps['duration'] >= 0).all(), "Duration must be non-negative"


def test_event_is_binary():
    tx = make_fake_transactions(n_users=50)
    gaps = compute_subscription_gaps(tx)
    assert set(gaps['event'].unique()).issubset({0, 1}), \
        "Event column must be binary (0 or 1)"


def test_kaplan_meier_fits():
    tx = make_fake_transactions(n_users=100)
    gaps = compute_subscription_gaps(tx)
    kmf  = fit_kaplan_meier(gaps)
    sf   = kmf.survival_function_
    col  = sf.columns[0]
    assert (sf[col] >= 0).all() and (sf[col] <= 1).all(), \
        "Survival function must be in [0, 1]"
    assert sf[col].iloc[0] <= 1.0, "Survival starts at or below 1"
    assert sf[col].is_monotonic_decreasing or \
           (sf[col].diff().dropna() <= 1e-10).all(), \
        "Survival function must be non-increasing"


# ── Data integrity tests ──────────────────────────────────────────────────────

def test_no_duplicate_users_in_rfm():
    tx = make_fake_transactions(n_users=50)
    rfm = build_rfm_features(tx)
    assert rfm['msno'].nunique() == len(rfm), \
        "RFM must have exactly one row per user"


def test_no_duplicate_users_in_subscription():
    tx = make_fake_transactions(n_users=50)
    sub = build_subscription_features(tx)
    assert sub['msno'].nunique() == len(sub), \
        "Subscription features must have one row per user"
