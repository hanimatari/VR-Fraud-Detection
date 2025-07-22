import streamlit as st
import pandas as pd
import pickle

# ─── 1) Load model & grab feature names ───────────────────────────────────
@st.cache_data
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
FEATURE_NAMES = list(model.feature_names_in_)

# ─── 2) Preprocessing (mirror your Colab logic) ───────────────────────────
def preprocess(raw_df):
    df = raw_df.copy()

    # Simulate market_value if missing
    if 'market_value' not in df.columns:
        df['market_value'] = df['price_paid'] * 1.0

    # Core features
    df['price_difference']       = df['price_paid'] - df['market_value']
    df['is_overpriced']          = (df['price_difference'] / df['market_value']) > 0.3
    df['user_transaction_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['is_repeating_user']      = df['user_transaction_count'] > 1
    df['is_withdrawal']          = df['type'].isin(['CASH_OUT','WITHDRAW'])
    df['suspicious_withdrawal']  = df['is_withdrawal'] & df['is_overpriced']

    # Drop unused IDs
    df = df.drop(columns=['user_id', 'nameDest'], errors='ignore')

    # One-hot encode 'type'
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # Ensure all model features are present
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training
    return df[FEATURE_NAMES]

# ─── 3) Annotate reasons & pack final DataFrame ───────────────────────────
def annotate_reasons(X, raw_df):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:,1].round(3)

    # Compute 95th-percentile cash-out threshold
    cash_threshold = X['price_paid'].quantile(0.95)

    reasons = []
    for i, row in X.iterrows():
        if y_pred[i] == 0:
            reasons.append("")  # not flagged
        elif row['suspicious_withdrawal']:
            reasons.append("Overpriced + withdrawal")
        elif row['is_overpriced']:
            reasons.append("Price > market by 30%")
        elif row['is_repeating_user']:
            reasons.append(f"Repeat user ({int(row['user_transaction_count'])} txns)")
        elif row['is_withdrawal'] and row['price_paid'] > cash_threshold:
            reasons.append(f"Large cash‐out (>p95: {int(cash_threshold)})")
        else:
            reasons.append("Other model signal")

    # Build annotated DataFrame
    annotated = raw_df.copy().reset_index(drop=True)
    annotated['is_fraud']         = y_pred
    annotated['fraud_prob']       = y_proba
    annotated['flag_reason']      = reasons
    annotated['market_value']     = X['market_value'].values
    annotated['price_difference'] = X['price_difference'].values

    return annotated

# ─── 4) Streamlit UI ─────────────────────────────────────────────────────
st.title("🚀 VR Fraud Detector Demo")
st.write("Upload your CSV of transactions; see fraud flags, reasons, and summaries.")

uploaded = st.file_uploader("Choose a CSV file", type="csv")
if uploaded:
    raw = pd.read_csv(uploaded)
    st.write("### Raw preview", raw.head())

    # 1) Engineer features
    X = preprocess(raw)
    engineered_cols = [
        'market_value','price_difference',
        'user_transaction_count','is_overpriced',
        'is_repeating_user','is_withdrawal','suspicious_withdrawal'
    ]
    st.write("### Engineered preview", X[engineered_cols].head())

    # 2) Annotate & predict
    results = annotate_reasons(X, raw)
    st.write("### Annotated results", results.head())

    # 3) Download annotated CSV
    csv = results.to_csv(index=False)
    st.download_button("⬇️ Download annotated CSV", csv,
                       file_name="vr_fraud_with_reasons.csv",
                       mime="text/csv")

    # ─── Visual summary ────────────────────────────────────────────────
    st.write("## Visual summary")

    # a) Flagged vs Normal
    raw_counts = results['is_fraud'].value_counts().sort_index()
    normal_cnt = int(raw_counts.get(0, 0))
    flagged_cnt = int(raw_counts.get(1, 0))
    counts_named = pd.Series({'Normal': normal_cnt, 'Flagged': flagged_cnt})
    st.write("### Transaction counts")
    st.bar_chart(counts_named)

    # b) Top 5 flag reasons (only non-empty)
    reason_series = results.loc[results['flag_reason'] != '', 'flag_reason']
    reason_counts = reason_series.value_counts()
    top5 = reason_counts.nlargest(5)
    other = reason_counts.iloc[5:].sum() if len(reason_counts) > 5 else 0
    reason_summary = pd.concat([top5, pd.Series({'Other': other})]) \
                       .sort_values(ascending=False)
    st.write("### Top 5 flag reasons")
    st.bar_chart(reason_summary)

    # c) Summary metrics table
    total   = normal_cnt + flagged_cnt
    flag_pct= round(100 * flagged_cnt / total, 1) if total else 0
    avg_prob= round(results['fraud_prob'].mean(), 3)
    summary_df = pd.DataFrame({
        "Metric": ["Total txns", "Flagged txns", "Flag rate (%)", "Avg fraud_prob"],
        "Value":  [ total,       flagged_cnt,   flag_pct,        avg_prob    ]
    })
    st.write("### Summary metrics")
    st.table(summary_df)
