
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from src.config import COLORS, USER_COL


# ── Core survival preparation ─────────────────────────────────────────────────

def compute_subscription_gaps(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.sort_values([USER_COL, 'transaction_date'])

    tx['next_transaction_date'] = tx.groupby(USER_COL)['transaction_date'].shift(-1)


    tx['gap_days'] = (
        tx['next_transaction_date'] - tx['membership_expire_date']
    ).dt.days


    tx['gap_days_clipped'] = tx['gap_days'].clip(lower=0)

    tx['event'] = tx['next_transaction_date'].notnull().astype(int)

    max_gap = tx['gap_days_clipped'].dropna().max()
    tx['duration'] = tx['gap_days_clipped'].fillna(max_gap)

    gaps = tx[[USER_COL, 'duration', 'event',
               'membership_expire_date', 'is_cancel']].copy()

    print(f"Subscription gaps: {len(gaps):,} periods")
    print(f"  Events (renewals observed): {gaps['event'].sum():,} "
          f"({gaps['event'].mean():.1%})")
    print(f"  Censored (last subscription): {(gaps['event']==0).sum():,}")
    print(f"  Duration range: {gaps['duration'].min():.0f} – "
          f"{gaps['duration'].max():.0f} days")

    return gaps


def fit_kaplan_meier(gaps: pd.DataFrame, label: str = 'All users') -> KaplanMeierFitter:

    kmf = KaplanMeierFitter(label=label)
    kmf.fit(
        durations=gaps['duration'],
        event_observed=gaps['event'],
    )
    return kmf


# ── Threshold derivation ──────────────────────────────────────────────────────
def derive_churn_threshold(kmf: KaplanMeierFitter,
                           thresholds: list = None) -> dict:
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    churn_prob = 1 - kmf.survival_function_
    col = churn_prob.columns[0]

    results = {}
    print("\nChurn probability thresholds:")
    print("-" * 35)
    for t in thresholds:
        mask = churn_prob[col] >= t
        if mask.any():
            day = churn_prob[mask].index.min()
            results[t] = int(day)
            print(f"  {t:.0%} churn probability: day {int(day)}")
        else:
            results[t] = None
            print(f"  {t:.0%} churn probability: not reached in data")


    survival = kmf.survival_function_.reset_index()
    survival.columns = ['timeline', 'survival']
    survival['drop'] = survival['survival'].diff().abs()
    elbow_day = survival.loc[survival['drop'].idxmax(), 'timeline']
    results['elbow'] = int(elbow_day)
    print(f"  Elbow point (largest drop): day {int(elbow_day)}")

    return results


# ── Segmented survival analysis ───────────────────────────────────────────────

def segment_by_frequency(gaps: pd.DataFrame,
                          labels: pd.DataFrame) -> pd.DataFrame:

    tx_counts = gaps.groupby(USER_COL)['event'].count().reset_index()
    tx_counts.columns = [USER_COL, 'n_subscriptions']

    def segment(n):
        if n == 1:   return 'one-time'
        if n <= 3:   return 'low (2-3)'
        if n <= 10:  return 'medium (4-10)'
        return 'high (10+)'

    tx_counts['segment'] = tx_counts['n_subscriptions'].apply(segment)
    gaps = gaps.merge(tx_counts[[USER_COL, 'segment']], on=USER_COL, how='left')
    return gaps


def compute_segment_thresholds(gaps: pd.DataFrame) -> pd.DataFrame:

    segments = gaps['segment'].unique()
    rows = []

    for seg in sorted(segments):
        seg_gaps = gaps[gaps['segment'] == seg]
        kmf = fit_kaplan_meier(seg_gaps, label=seg)
        thresholds = derive_churn_threshold(kmf, thresholds=[0.5])
        rows.append({
            'segment':          seg,
            'n_periods':        len(seg_gaps),
            'n_users':          seg_gaps[USER_COL].nunique(),
            'median_duration':  seg_gaps['duration'].median(),
            'churn_day_50pct':  thresholds.get(0.5),
        })

    summary = pd.DataFrame(rows).sort_values('churn_day_50pct')
    print("\nSegment-level churn thresholds:")
    print(summary.to_string(index=False))
    return summary


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_survival_overview(kmf: KaplanMeierFitter,
                            thresholds: dict,
                            segment_gaps: pd.DataFrame = None):
    """
    Two-panel survival analysis plot:
    Left:  overall churn probability curve with threshold annotations
    Right: segmented survival curves (if segment data provided)
    """
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)


    ax1 = fig.add_subplot(gs[0])
    churn_prob = 1 - kmf.survival_function_

    col = churn_prob.columns[0]
    ax1.plot(churn_prob.index, churn_prob[col],
             color=COLORS['primary'], linewidth=2.5, label='Churn probability')
    ax1.fill_between(churn_prob.index, churn_prob[col],
                     alpha=0.1, color=COLORS['primary'])
    
    

    colors_t = [COLORS['accent'], COLORS['danger'], COLORS['secondary']]
    for (prob, day), col in zip(thresholds.items(), colors_t):
        if day and prob != 'elbow':
            ax1.axhline(prob, linestyle='--', color=col, alpha=0.7, linewidth=1.5)
            ax1.axvline(day, linestyle='--', color=col, alpha=0.7, linewidth=1.5,
                        label=f'{prob:.0%} churn = day {day}')

    ax1.set_xlabel('Days Since Last Subscription Expiry', fontsize=12)
    ax1.set_ylabel('Churn Probability', fontsize=12)
    ax1.set_title('Data-Driven Churn Threshold\n(Kaplan-Meier Estimator)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 1)

    # ── Right: segmented survival curves ─────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    if segment_gaps is not None:
        seg_colors = [COLORS['primary'], COLORS['secondary'],
                      COLORS['accent'], COLORS['danger']]
        segments = sorted(segment_gaps['segment'].unique())

        for seg, col in zip(segments, seg_colors):
            seg_data = segment_gaps[segment_gaps['segment'] == seg]
            kmf_seg  = fit_kaplan_meier(seg_data, label=seg)
            surv = 1 - kmf_seg.survival_function_
            col  = surv.columns[0]
            ax2.plot(surv.index, surv[col], linewidth=2, label=seg)


        ax2.axhline(0.5, linestyle='--', color='gray', alpha=0.5,
                    label='50% churn level')
        ax2.set_xlabel('Days Since Last Subscription Expiry', fontsize=12)
        ax2.set_ylabel('Churn Probability', fontsize=12)
        ax2.set_title('Churn Probability by User Segment\n'
                       '(one-time vs repeat subscribers)', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 180)
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/survival_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/survival_analysis.png")


def plot_segment_thresholds(summary: pd.DataFrame):
    """Bar chart of 50% churn threshold by user segment."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(summary['segment'], summary['churn_day_50pct'],
                   color=COLORS['primary'], alpha=0.8)

    for bar, val in zip(bars, summary['churn_day_50pct']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'Day {val:.0f}', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Days to 50% Churn Probability', fontsize=12)
    ax.set_title('Optimal Churn Threshold by User Segment', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, summary['churn_day_50pct'].max() * 1.2)

    plt.tight_layout()
    plt.savefig('results/segment_thresholds.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/segment_thresholds.png")
