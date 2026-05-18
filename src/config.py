from pathlib import Path
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PRO = ROOT / 'data' / 'processed'
RESULTS  = ROOT / 'results'

TRAIN_PATH        = DATA_RAW / 'train_v2.csv'
MEMBERS_PATH      = DATA_RAW / 'members_v3.csv'
TRANSACTIONS_PATH = DATA_RAW / 'transactions_v2.csv'
TRANSACTIONS_V1_PATH = DATA_RAW / 'transactions.csv'

# ── Dataset constants ────────────────────────────────────────────────
CHURN_COL    = 'is_churn'
USER_COL     = 'msno'
RANDOM_STATE = 42
CHUNK_SIZE   = 500_000

FEATURE_CUTOFF = pd.Timestamp('2017-02-28')

# ── Survival analysis ────────────────────────────────────────────────
SURVIVAL_THRESHOLDS = [0.3, 0.5, 0.7]

# ── Model ────────────────────────────────────────────────────────────
CV_FOLDS  = 5
TEST_SIZE = 0.2

XGB_PARAMS = {
    'learning_rate':    0.05,
    'max_depth':        6,
    'n_estimators':     1000,
    'min_child_weight': 10,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'scale_pos_weight': 10,
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'tree_method':      'hist',
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
    'early_stopping_rounds': 50,
}

LGB_PARAMS = {
    'learning_rate':    0.05,
    'max_depth':        6,
    'n_estimators':     1000,
    'num_leaves':       64,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'is_unbalance':     True,
    'objective':        'binary',
    'metric':           'auc',
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
    'verbose':          -1,
}

# ── Visualization ────────────────────────────────────────────────────
COLORS = {
    'primary':   '#2E86AB',
    'secondary': '#A23B72',
    'accent':    '#F18F01',
    'danger':    '#C73E1D',
    'neutral':   '#3B3B3B',
}