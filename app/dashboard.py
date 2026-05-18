# app/dashboard.py
"""
KKBOX Churn Prediction Dashboard
For marketing and product teams — not a model demo.

Three modes:
1. Campaign Builder  — score users, export prioritized lists
2. Cohort Explorer   — slice churn risk by segment
3. Model Explainer   — understand model performance and thresholds
"""
import pickle
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (roc_curve, precision_recall_curve,
                              roc_auc_score, average_precision_score)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#2E86AB',
    'danger':    '#C73E1D',
    'warning':   '#F18F01',
    'success':   '#3BB273',
    'neutral':   '#6C757D',
    'bg_dark':   '#1E1E2E',
    'bg_card':   '#F8F9FA',
}

RISK_COLORS = {
    'Critical':  COLORS['danger'],
    'High':      COLORS['warning'],
    'Medium':    '#E8A838',
    'Low':       COLORS['success'],
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifact_path = Path('results/artifacts.pkl')
    if not artifact_path.exists():
        return None
    with open(artifact_path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_training_data():
    """Load pre-processed feature matrix for cohort analysis."""
    path = Path('results/feature_matrix.parquet')
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── Utility functions ─────────────────────────────────────────────────────────

def score_to_risk_segment(scores: np.ndarray, threshold: float) -> pd.Series:
    """Convert raw churn scores to business risk segments."""
    segments = pd.cut(
        scores,
        bins=[0, 0.3, 0.5, threshold, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True,
    )
    return segments


def get_intervention(risk: str) -> str:
    """Map risk segment to recommended intervention."""
    mapping = {
        'Critical': '🚨 Immediate outreach — personal call or high-value offer',
        'High':     '⚠️  Targeted campaign — discount or feature highlight',
        'Medium':   '📧 Nurture sequence — engagement email or push notification',
        'Low':      '✅ Monitor — no immediate action required',
    }
    return mapping.get(risk, 'Unknown')


def predict_from_features(X: pd.DataFrame, fold_results: list) -> np.ndarray:
    """Run ensemble inference using last fold models."""
    last = fold_results[-1]
    feature_names = last['feature_names']

    # Align columns
    missing = set(feature_names) - set(X.columns)
    for col in missing:
        X[col] = 0
    X = X[feature_names].fillna(0)

    w_xgb = last['xgb_weight']
    w_lgb = last['lgb_weight']

    preds_xgb = last['xgb_model'].predict(X)
    preds_lgb = last['lgb_model'].predict_proba(X)[:, 1]

    return (w_xgb * preds_xgb + w_lgb * preds_lgb) / (w_xgb + w_lgb)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(artifacts):
    st.sidebar.image(
        "https://img.icons8.com/color/96/combo-chart.png", width=60
    )
    st.sidebar.title("Churn Intelligence")
    st.sidebar.caption("Powered by XGBoost + LightGBM ensemble")

    mode = st.sidebar.radio(
        "Select Mode",
        ["🎯 Campaign Builder", "🔍 Cohort Explorer", "📈 Model Explainer"],
        index=0,
    )

    st.sidebar.divider()

    # Global threshold slider
    default_threshold = artifacts['threshold'] if artifacts else 0.5
    threshold = st.sidebar.slider(
        "Churn Probability Threshold",
        min_value=0.1,
        max_value=0.99,
        value=float(default_threshold),
        step=0.01,
        help="Users above this threshold are flagged as churners. "
             "Lower = catch more churners (higher recall, lower precision). "
             "Higher = fewer false alarms (lower recall, higher precision).",
    )

    # Live precision/recall at current threshold
    if artifacts and 'oof_preds' in artifacts:
        oof   = artifacts['oof_preds']
        y     = artifacts['y_true']
        prec  = np.mean(y[oof >= threshold]) if (oof >= threshold).any() else 0
        rec   = np.sum((oof >= threshold) & (y == 1)) / max(y.sum(), 1)
        flagged = (oof >= threshold).mean()

        col1, col2 = st.sidebar.columns(2)
        col1.metric("Precision", f"{prec:.1%}")
        col2.metric("Recall", f"{rec:.1%}")
        st.sidebar.metric(
            "Users Flagged",
            f"{flagged:.1%}",
            help="Fraction of total users who would be contacted"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Model: XGBoost + LightGBM ensemble  \n"
        "Dataset: KKBOX (970K users, 16M transactions)  \n"
        "OOF AUC: 0.9171 | AP: 0.7174"
    )

    return mode, threshold


# ── Mode 1: Campaign Builder ──────────────────────────────────────────────────

def render_campaign_builder(artifacts, threshold):
    st.title("🎯 Campaign Builder")
    st.caption(
        "Score your users, segment by risk level, and export prioritized "
        "contact lists for your CRM or marketing automation tool."
    )

    # ── Option A: Use training data ───────────────────────────────────
    tab1, tab2 = st.tabs(["📂 Use Training Data", "⬆️ Upload New Users"])

    with tab1:
        st.markdown("**Score users from the KKBOX training dataset.**")

        if artifacts is None:
            st.error("No artifacts found. Run `python -m scripts.train` first.")
            return

        oof_preds = artifacts['oof_preds']
        y_true    = artifacts['y_true'].values

        # Build scored dataframe
        scored_df = pd.DataFrame({
            'churn_score':  oof_preds,
            'actual_churn': y_true,
        })
        scored_df['risk_segment'] = score_to_risk_segment(oof_preds, threshold)
        scored_df['flagged']      = oof_preds >= threshold
        scored_df['intervention'] = scored_df['risk_segment'].map(
            lambda r: get_intervention(str(r))
        )
        scored_df['user_id'] = [f"USER_{i:07d}" for i in range(len(scored_df))]

        render_campaign_summary(scored_df, threshold)
        render_user_table(scored_df, threshold)

    with tab2:
        render_upload_scorer(artifacts, threshold)


def render_campaign_summary(scored_df, threshold):
    """KPI cards + risk breakdown for the scored dataset."""
    flagged     = scored_df[scored_df['flagged']]
    n_flagged   = len(flagged)
    n_total     = len(scored_df)
    precision   = flagged['actual_churn'].mean() if len(flagged) > 0 else 0
    recall      = (
        flagged['actual_churn'].sum() / max(scored_df['actual_churn'].sum(), 1)
    )

    st.subheader("Campaign Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users",        f"{n_total:,}")
    c2.metric("Users to Contact",   f"{n_flagged:,}")
    c3.metric("Contact Rate",       f"{n_flagged/n_total:.1%}")
    c4.metric("Expected Precision", f"{precision:.1%}",
              help="Of flagged users, % who will actually churn")
    c5.metric("Churn Recall",       f"{recall:.1%}",
              help="% of all churners who will be caught")

    st.divider()

    # Risk segment breakdown
    st.subheader("Risk Segment Breakdown")
    seg_counts = (
        scored_df.groupby('risk_segment', observed=True)
        .agg(
            users=('churn_score', 'count'),
            avg_score=('churn_score', 'mean'),
            actual_churn_rate=('actual_churn', 'mean'),
        )
        .reset_index()
    )

    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        fig = px.bar(
            seg_counts,
            x='risk_segment',
            y='users',
            color='risk_segment',
            color_discrete_map=RISK_COLORS,
            text='users',
            title='Users by Risk Segment',
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=350,
                          xaxis_title='Risk Segment',
                          yaxis_title='Number of Users')
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Segment Details**")
        display_seg = seg_counts.copy()
        display_seg['avg_score']          = display_seg['avg_score'].map('{:.3f}'.format)
        display_seg['actual_churn_rate']  = display_seg['actual_churn_rate'].map('{:.1%}'.format)
        display_seg['users']              = display_seg['users'].map('{:,}'.format)
        display_seg.columns = ['Segment', 'Users', 'Avg Score', 'Actual Churn Rate']
        st.dataframe(display_seg, hide_index=True, use_container_width=True)

        st.markdown("**Recommended Actions**")
        for seg in ['Critical', 'High', 'Medium', 'Low']:
            st.markdown(f"**{seg}:** {get_intervention(seg)}")


def render_user_table(scored_df, threshold):
    """Filterable, downloadable user table."""
    st.divider()
    st.subheader("User List")

    col1, col2, col3 = st.columns(3)
    with col1:
        seg_filter = st.multiselect(
            "Filter by Risk Segment",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High'],
        )
    with col2:
        score_min = st.slider("Min Churn Score", 0.0, 1.0,
                               float(threshold), 0.01)
    with col3:
        sort_by = st.selectbox("Sort By",
                                ['churn_score', 'risk_segment'])

    filtered = scored_df[
        scored_df['risk_segment'].astype(str).isin(seg_filter) &
        (scored_df['churn_score'] >= score_min)
    ].sort_values(sort_by, ascending=False)

    st.caption(f"Showing {len(filtered):,} users")

    display_cols = ['user_id', 'churn_score', 'risk_segment', 'intervention']
    display = filtered[display_cols].copy()
    display['churn_score'] = display['churn_score'].round(4)
    display.columns = ['User ID', 'Churn Score', 'Risk Segment', 'Recommended Action']

    st.dataframe(
        display.head(500),
        hide_index=True,
        use_container_width=True,
        column_config={
            'Churn Score': st.column_config.ProgressColumn(
                'Churn Score', min_value=0, max_value=1, format='%.3f'
            ),
        }
    )

    # Download button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download {len(filtered):,} Users as CSV",
        data=csv,
        file_name=f"churn_campaign_threshold_{threshold:.2f}.csv",
        mime='text/csv',
    )


def render_upload_scorer(artifacts, threshold):
    """Score a user-uploaded CSV of new users."""
    st.markdown("""
    Upload a CSV with user transaction features. Required columns:
    `recency_days`, `frequency`, `total_paid`, `avg_paid`, `tenure_days`,
    `days_to_expiry`, `cancel_rate`, `auto_renew_rate`, `discount_rate`

    Optional: any additional columns will be passed through to the output.
    """)

    uploaded = st.file_uploader("Upload user feature CSV", type=['csv'])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} users")

            scores = predict_from_features(
                df.copy(), artifacts['fold_results']
            )
            df['churn_score']    = scores
            df['risk_segment']   = score_to_risk_segment(scores, threshold)
            df['intervention']   = df['risk_segment'].map(
                lambda r: get_intervention(str(r))
            )
            df['flagged']        = scores >= threshold

            render_campaign_summary(df, threshold)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Scored Users",
                data=csv,
                file_name="churn_scored_users.csv",
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ── Mode 2: Cohort Explorer ───────────────────────────────────────────────────

def render_cohort_explorer(artifacts, threshold):
    st.title("🔍 Cohort Explorer")
    st.caption(
        "Understand which user segments drive churn. "
        "Use this to inform product decisions, not just marketing campaigns."
    )

    if artifacts is None:
        st.error("No artifacts found.")
        return

    oof_preds = artifacts['oof_preds']
    y_true    = artifacts['y_true'].values

    # Rebuild scored df with synthetic cohort features
    # In production this would join back to member/transaction data
    n = len(oof_preds)
    np.random.seed(42)

    cohort_df = pd.DataFrame({
        'churn_score':  oof_preds,
        'actual_churn': y_true,
        'flagged':      oof_preds >= threshold,
    })

    # ── Churn score distribution ──────────────────────────────────────
    st.subheader("Churn Score Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=oof_preds[y_true == 0],
            name='Non-Churners',
            opacity=0.7,
            marker_color=COLORS['primary'],
            nbinsx=50,
            histnorm='density',
        ))
        fig.add_trace(go.Histogram(
            x=oof_preds[y_true == 1],
            name='Churners',
            opacity=0.7,
            marker_color=COLORS['danger'],
            nbinsx=50,
            histnorm='density',
        ))
        fig.add_vline(
            x=threshold, line_dash='dash', line_color='black',
            annotation_text=f'Threshold {threshold:.2f}',
        )
        fig.update_layout(
            barmode='overlay',
            title='Score Distribution by True Label',
            xaxis_title='Churn Probability',
            yaxis_title='Density',
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Score decile analysis
        cohort_df['decile'] = pd.qcut(
            cohort_df['churn_score'], q=10,
            labels=[f'D{i}' for i in range(1, 11)]
        )
        decile_stats = cohort_df.groupby('decile', observed=True).agg(
            n_users=('churn_score', 'count'),
            avg_score=('churn_score', 'mean'),
            churn_rate=('actual_churn', 'mean'),
        ).reset_index()

        fig = px.bar(
            decile_stats,
            x='decile',
            y='churn_rate',
            color='churn_rate',
            color_continuous_scale=['#3BB273', '#F18F01', '#C73E1D'],
            title='Actual Churn Rate by Score Decile',
            text=decile_stats['churn_rate'].map('{:.1%}'.format),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=350,
            xaxis_title='Score Decile (D1=Lowest Risk)',
            yaxis_title='Actual Churn Rate',
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Threshold sensitivity ─────────────────────────────────────────
    st.divider()
    st.subheader("Threshold Sensitivity Analysis")
    st.caption(
        "How does your choice of threshold affect campaign size and effectiveness? "
        "Use this to align with your marketing budget."
    )

    thresholds = np.arange(0.1, 1.0, 0.02)
    rows = []
    for t in thresholds:
        flagged = oof_preds >= t
        if flagged.sum() == 0:
            continue
        rows.append({
            'threshold':   t,
            'n_flagged':   flagged.sum(),
            'contact_pct': flagged.mean(),
            'precision':   y_true[flagged].mean(),
            'recall':      y_true[flagged].sum() / max(y_true.sum(), 1),
        })
    sens_df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens_df['threshold'], y=sens_df['precision'],
            name='Precision', line=dict(color=COLORS['success'], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=sens_df['threshold'], y=sens_df['recall'],
            name='Recall', line=dict(color=COLORS['danger'], width=2)
        ))
        fig.add_vline(
            x=threshold, line_dash='dash', line_color='black',
            annotation_text='Current',
        )
        fig.update_layout(
            title='Precision vs Recall at Each Threshold',
            xaxis_title='Threshold',
            yaxis_title='Score',
            yaxis=dict(tickformat='.0%'),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens_df['threshold'],
            y=sens_df['contact_pct'],
            fill='tozeroy',
            line=dict(color=COLORS['primary'], width=2),
            name='Contact Rate',
        ))
        fig.add_vline(
            x=threshold, line_dash='dash', line_color='black',
            annotation_text='Current',
        )
        fig.update_layout(
            title='Campaign Size at Each Threshold',
            xaxis_title='Threshold',
            yaxis_title='% Users Contacted',
            yaxis=dict(tickformat='.0%'),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Current threshold details
    current = sens_df[sens_df['threshold'].round(2) == round(threshold, 2)]
    if len(current) > 0:
        row = current.iloc[0]
        st.info(
            f"**At threshold {threshold:.2f}:** "
            f"Contact {row['n_flagged']:,.0f} users ({row['contact_pct']:.1%} of base) — "
            f"Precision {row['precision']:.1%} — "
            f"Recall {row['recall']:.1%}"
        )

    # ── Business impact estimator ─────────────────────────────────────
    st.divider()
    st.subheader("💰 Business Impact Estimator")
    st.caption("Estimate the revenue impact of your retention campaign.")

    col1, col2, col3 = st.columns(3)
    with col1:
        arpu = st.number_input(
            "Monthly Revenue per User ($)", value=10, min_value=1
        )
    with col2:
        campaign_cost = st.number_input(
            "Cost per Contacted User ($)", value=2, min_value=0
        )
    with col3:
        retention_rate = st.slider(
            "Campaign Success Rate", 0.05, 0.50, 0.15,
            help="% of flagged churners who are retained by the campaign"
        )

    if len(current) > 0:
        row          = current.iloc[0]
        n_flagged    = int(row['n_flagged'])
        true_churners = int(n_flagged * row['precision'])
        retained      = int(true_churners * retention_rate)
        revenue_saved = retained * arpu * 12    # annual
        campaign_spend = n_flagged * campaign_cost
        net_value      = revenue_saved - campaign_spend
        roi            = (net_value / max(campaign_spend, 1)) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users Contacted",    f"{n_flagged:,}")
        c2.metric("Churners Retained",  f"{retained:,}")
        c3.metric("Annual Revenue Saved", f"${revenue_saved:,.0f}")
        c4.metric("Net ROI",            f"{roi:.0f}%",
                  delta="positive" if net_value > 0 else "negative")

        if net_value > 0:
            st.success(
                f"Campaign generates **${net_value:,.0f}** net value "
                f"after ${campaign_spend:,.0f} spend."
            )
        else:
            st.warning(
                f"Campaign costs **${abs(net_value):,.0f}** more than it saves. "
                f"Consider raising the threshold or negotiating lower contact costs."
            )


# ── Mode 3: Model Explainer ───────────────────────────────────────────────────

def render_model_explainer(artifacts, threshold):
    st.title("📈 Model Explainer")
    st.caption(
        "Understand model performance, feature drivers, and validation methodology. "
        "For DS and leadership audiences."
    )

    if artifacts is None:
        st.error("No artifacts found.")
        return

    oof_preds    = artifacts['oof_preds']
    y_true       = artifacts['y_true'].values
    fold_results = artifacts['fold_results']

    # ── Top metrics ───────────────────────────────────────────────────
    auc = roc_auc_score(y_true, oof_preds)
    ap  = average_precision_score(y_true, oof_preds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOF AUC",              f"{auc:.4f}")
    c2.metric("Average Precision",    f"{ap:.4f}")
    c3.metric("Baseline AP",          f"{y_true.mean():.4f}",
              help="AP if you randomly predicted churn rate for everyone")
    c4.metric("Lift over Baseline",   f"{ap/y_true.mean():.1f}x")

    # ── ROC + PR curves ───────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fpr, tpr, _ = roc_curve(y_true, oof_preds)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'Model (AUC={auc:.4f})',
            line=dict(color=COLORS['primary'], width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            line=dict(color='gray', width=1, dash='dash'),
        ))

        # Mark threshold point
        thresh_idx = np.argmin(np.abs(np.sort(oof_preds)[::-1] - threshold))
        flagged    = oof_preds >= threshold
        fp_rate    = np.sum((flagged) & (y_true == 0)) / max((y_true == 0).sum(), 1)
        tp_rate    = np.sum((flagged) & (y_true == 1)) / max(y_true.sum(), 1)
        fig.add_trace(go.Scatter(
            x=[fp_rate], y=[tp_rate],
            mode='markers',
            marker=dict(size=12, color=COLORS['danger'], symbol='x'),
            name=f'Threshold {threshold:.2f}',
        ))

        fig.update_layout(
            title='ROC Curve (Out-of-Fold)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prec, rec, _ = precision_recall_curve(y_true, oof_preds)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rec, y=prec,
            name=f'Model (AP={ap:.4f})',
            line=dict(color=COLORS['secondary'] if 'secondary' in COLORS
                      else COLORS['primary'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(46,134,171,0.1)',
        ))
        fig.add_hline(
            y=y_true.mean(),
            line_dash='dash', line_color='gray',
            annotation_text=f'Baseline ({y_true.mean():.1%})',
        )
        fig.update_layout(
            title='Precision-Recall Curve (Out-of-Fold)',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── CV fold stability ─────────────────────────────────────────────
    st.divider()
    st.subheader("Cross-Validation Stability")

    fold_data = pd.DataFrame([
        {
            'Fold':     f"Fold {r['fold']}",
            'XGBoost':  r['xgb_auc'],
            'LightGBM': r['lgb_auc'],
            'Ensemble': r['ensemble_auc'],
        }
        for r in fold_results
    ])

    fig = go.Figure()
    for model, color in [
        ('XGBoost',  COLORS['primary']),
        ('LightGBM', COLORS['warning']),
        ('Ensemble', COLORS['success']),
    ]:
        fig.add_trace(go.Bar(
            x=fold_data['Fold'],
            y=fold_data[model],
            name=model,
            marker_color=color,
        ))

    fig.update_layout(
        barmode='group',
        title='AUC by Model and Fold',
        yaxis=dict(range=[0.90, 0.93]),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    stats = fold_data.set_index('Fold').agg(['mean', 'std', 'min', 'max'])
    st.dataframe(
        stats.T.style.format('{:.5f}'),
        use_container_width=True,
    )

    # ── Feature importance ────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Importance")
    st.caption(
        "LightGBM gain importance averaged across all 5 folds. "
        "Gain measures how much each feature reduces the loss — "
        "not just how often it is used."
    )

    feature_names = fold_results[0]['feature_names']
    avg_importance = np.mean([
        r['lgb_model'].feature_importances_ for r in fold_results
    ], axis=0)

    imp_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': avg_importance,
    }).sort_values('Importance', ascending=True).tail(20)

    fig = px.bar(
        imp_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#AED6F1', COLORS['primary']],
        title='Top 20 Features by Gain Importance',
    )
    fig.update_layout(
        height=550,
        coloraxis_showscale=False,
        yaxis_title='',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Methodology note ──────────────────────────────────────────────
    st.divider()
    st.subheader("Methodology Notes")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Why stratified k-fold, not time-series CV?**

        Unlike the Optiver stock prediction project, churn labels here are
        defined for a specific evaluation period (March 2017) and features
        are computed from historical transactions (pre-February 2017).
        There is no temporal leakage risk — the feature/label split is
        already enforced by the cutoff date. Stratified k-fold preserves
        the 9% churn rate in every fold.

        **Why AUC + Average Precision?**

        With 9% churn rate, accuracy is misleading (91% accuracy by
        predicting nobody churns). AUC measures ranking quality across
        all thresholds. Average Precision summarizes the precision-recall
        curve — more informative than AUC for imbalanced problems because
        it weights performance on the minority class more heavily.
        """)

    with col2:
        st.markdown("""
        **Why threshold 0.835?**

        The default 0.5 threshold is wrong for 9% churn rate. A model
        outputting 0.4 for a likely churner would be missed. The optimal
        threshold is tuned to achieve 60% recall (catch 60% of churners)
        at maximum precision — a reasonable trade-off for retention
        campaigns where each outreach has a cost.

        **Data leakage prevention:**

        All features are computed from transactions with
        `transaction_date ≤ 2017-02-28`. The churn label evaluates
        March 2017 behavior. This ensures no future information leaks
        into model training. A naive model using post-cutoff transactions
        achieved AUC 0.9958 — a clear leakage signal that was caught
        and corrected.
        """)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    artifacts = load_artifacts()

    if artifacts is None:
        st.error(
            "⚠️ No model artifacts found. "
            "Run `python -m scripts.train` to train the model first."
        )
        st.stop()

    # Ensure y_true is a Series
    if not isinstance(artifacts.get('y_true'), pd.Series):
        artifacts['y_true'] = pd.Series(artifacts['y_true'])

    mode, threshold = render_sidebar(artifacts)

    if mode == "🎯 Campaign Builder":
        render_campaign_builder(artifacts, threshold)
    elif mode == "🔍 Cohort Explorer":
        render_cohort_explorer(artifacts, threshold)
    elif mode == "📈 Model Explainer":
        render_model_explainer(artifacts, threshold)


if __name__ == "__main__":
    main()
