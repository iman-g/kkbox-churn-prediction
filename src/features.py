
import numpy as np
import pandas as pd

from src.config import USER_COL, FEATURE_CUTOFF


# ── Transaction-based features ────────────────────────────────────────────────

def build_rfm_features(tx: pd.DataFrame,
                        reference_date: pd.Timestamp = None) -> pd.DataFrame:

    tx = tx[tx['transaction_date'] <= FEATURE_CUTOFF]
    if reference_date is None:
        reference_date = tx['transaction_date'].max()

    rfm = tx.groupby(USER_COL).agg(
        recency_days    = ('transaction_date',
                           lambda x: (reference_date - x.max()).days),
        frequency       = ('transaction_date', 'count'),
        total_paid      = ('actual_amount_paid', 'sum'),
        avg_paid        = ('actual_amount_paid', 'mean'),
        max_paid        = ('actual_amount_paid', 'max'),
        min_paid        = ('actual_amount_paid', 'min'),
    ).reset_index()

    print(f"RFM features: {rfm.shape}")
    return rfm


def build_subscription_features(tx: pd.DataFrame) -> pd.DataFrame:

    tx = tx[tx['transaction_date'] <= FEATURE_CUTOFF]
    tx = tx.sort_values([USER_COL, 'transaction_date'])

    sub = tx.groupby(USER_COL).agg(
        # Plan characteristics
        avg_plan_days       = ('payment_plan_days', 'mean'),
        most_common_plan    = ('payment_plan_days', lambda x: x.mode()[0]),
        plan_variety        = ('payment_plan_days', 'nunique'),

        # Price signals
        avg_list_price      = ('plan_list_price', 'mean'),
        avg_actual_price    = ('actual_amount_paid', 'mean'),

        # Cancellation behavior
        n_cancellations     = ('is_cancel', 'sum'),
        cancel_rate         = ('is_cancel', 'mean'),

        # Auto-renewal
        auto_renew_rate     = ('is_auto_renew', 'mean'),
        last_auto_renew     = ('is_auto_renew', 'last'),

        # Tenure
        first_transaction   = ('transaction_date', 'min'),
        last_transaction    = ('transaction_date', 'max'),
        last_expire         = ('membership_expire_date', 'max'),

        # Payment method diversity
        n_payment_methods   = ('payment_method_id', 'nunique'),
        last_payment_method = ('payment_method_id', 'last'),
    ).reset_index()

    # Derived: tenure in days
    sub['tenure_days'] = (
        sub['last_transaction'] - sub['first_transaction']
    ).dt.days.clip(lower=0)

    # Derived: days until expiry from last transaction date
    sub['days_to_expiry'] = (
        sub['last_expire'] - sub['last_transaction']
    ).dt.days

    # Derived: discount rate (how much below list price they paid)
    sub['discount_rate'] = (
        1 - sub['avg_actual_price'] / (sub['avg_list_price'] + 1e-6)
    ).clip(0, 1)

    # Drop raw date columns
    sub = sub.drop(columns=['first_transaction', 'last_transaction', 'last_expire'])

    print(f"Subscription features: {sub.shape}")
    return sub


# ── Member profile features ───────────────────────────────────────────────────

def build_member_features(members: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode member demographics.
    """
    mem = members.copy()

    # Age: clip impossible values, flag missing
    mem['age']         = mem['bd'].clip(10, 80)
    mem['age_missing'] = (mem['bd'] <= 0) | (mem['bd'] > 100)
    mem['age_missing'] = mem['age_missing'].astype(int)
    mem = mem.drop(columns=['bd'])

    # Gender: encode as binary + missing flag
    mem['gender_female']  = (mem['gender'] == 'female').astype(int)
    mem['gender_missing'] = (mem['gender'] == 'unknown').astype(int)
    mem = mem.drop(columns=['gender'])

    # Registration: extract year and compute account age
    mem['reg_year'] = pd.to_datetime(
        mem['registration_init_time'].astype(str),
        format='%Y%m%d', errors='coerce'
    ).dt.year
    mem['account_age_days'] = (
        pd.Timestamp('2017-03-31') -
        pd.to_datetime(mem['registration_init_time'].astype(str),
                       format='%Y%m%d', errors='coerce')
    ).dt.days.clip(lower=0)
    mem = mem.drop(columns=['registration_init_time'])

    print(f"Member features: {mem.shape}")
    return mem


# ── Master feature table ──────────────────────────────────────────────────────

def build_feature_matrix(labels: pd.DataFrame,
                          members: pd.DataFrame,
                          tx: pd.DataFrame) -> pd.DataFrame:

    print("Building feature matrix...")

    rfm  = build_rfm_features(tx)
    sub  = build_subscription_features(tx)
    mem  = build_member_features(members)

    # Start from labels (defines the universe of users)
    df = labels.copy()

    # Join features
    df = df.merge(rfm, on=USER_COL, how='left')
    df = df.merge(sub, on=USER_COL, how='left')
    df = df.merge(mem, on=USER_COL, how='left')

    # Drop user ID — not a feature
    df = df.drop(columns=[USER_COL])

    # Fill remaining nulls with 0 (users with no member profile)
    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c != 'is_churn']
    X = df[feature_cols]
    y = df['is_churn']




    mask_2017_jan_feb = (tx['transaction_date'] >= '2017-01-01') & \
                    (tx['transaction_date'] <= '2017-02-28')
    print(f"Jan-Feb 2017 transactions: {mask_2017_jan_feb.sum():,}")
    print(f"Users covered: {tx[mask_2017_jan_feb]['msno'].nunique():,}")


    print(f"Feature matrix: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Null check: {X.isnull().sum().sum()} remaining nulls")

    return X, y, feature_cols