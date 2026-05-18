import pandas as pd
import numpy as np
from src.config import (TRAIN_PATH, MEMBERS_PATH, 
                        TRANSACTIONS_PATH, TRANSACTIONS_V1_PATH,
                        USER_COL, CHUNK_SIZE, FEATURE_CUTOFF)


def parse_date(series: pd.Series) -> pd.Series:

    return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    print(f"Labels: {df.shape} | Churn rate: {df['is_churn'].mean():.2%}")
    return df


def load_members(users: set = None) -> pd.DataFrame:
    df = pd.read_csv(MEMBERS_PATH)
    if users:
        df = df[df[USER_COL].isin(users)]
    df['registration_init_time'] = parse_date(df['registration_init_time'])
    df['gender'] = df['gender'].fillna('unknown')
    df['bd'] = df['bd'].clip(0, 100)   # age: remove impossible values
    df['bd'] = df['bd'].replace(0, np.nan)
    print(f"Members: {df.shape}")
    return df


def load_transactions(users: set = None) -> pd.DataFrame:

    all_chunks = []
    
    for path in [TRANSACTIONS_V1_PATH, TRANSACTIONS_PATH]:
        if not path.exists():
            print(f"  Skipping {path.name} — not found")
            continue
        print(f"  Loading {path.name}...")
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            if users:
                chunk = chunk[chunk[USER_COL].isin(users)]
            chunk['transaction_date'] = parse_date(chunk['transaction_date'])
            chunk['membership_expire_date'] = parse_date(
                chunk['membership_expire_date']
            )
            all_chunks.append(chunk)
            print(f"    Chunk loaded: {len(chunk):,} rows kept")
    
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates()
    df = df.sort_values([USER_COL, 'transaction_date'])
    print(f"Transactions combined: {df.shape}")
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    labels   = load_labels()
    users    = set(labels[USER_COL])
    members  = load_members(users)
    tx       = load_transactions(users)
    return labels, members, tx