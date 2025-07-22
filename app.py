import streamlit as st
import pandas as pd
import pickle

# ─── 1) Load the trained RF model ────────────────────────────────────────
@st.cache_data
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
FEATURE_NAMES = list(model.feature_names_in_)

# ─── 2) Preprocessing ───────────────────────────────────────────────────
def preprocess(raw_df):
    df = raw_df.copy()
    if 'market_value' not in df.columns:
        df['market_value'] = df['price_paid'] * 1.0
    df['price_difference'] = df['price_paid'] - df['market_value']
    df['is_overpriced']    = (df['price_difference'] / df['market_value']) > 0.3
    df['user_transaction_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['is_repeating_user']      = df['user_transaction_count'] > 1
    df['is_withdrawal']         = df['type'].isin(['CASH_OUT', 'WITHDRAW'])
    df['suspicious_withdrawal'] = df['is_withdrawal'] & df['is_overpriced']
    df = df.drop(columns=['user_id','nameDest'], errors='ignore')
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_NAMES]

# ─── 3) Annotate reasons ─────────────────────────────────────────────────
def annotate_reasons(X, raw_df):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:,1].round(3)
    df_flags = X.copy()
    df_flags['is_fraud']   = y_pred
    df_flags['fraud_prob'] = y_proba

    reasons = []
    for _, row in df_flags.iterrows():
        if row['suspicious_withdrawal']:
            reasons.append("Overpriced + withdrawal")
        elif row['is_overpriced']:
            reasons.append("Price > market by 30%")
        elif row['is_repeating_user']:
            reasons.append(f"Repeat user ({int(row['user_transaction_count'])} txns)")
        else:
            reasons.append("No rule triggered")
    df_flags['flag_reason'] = reasons

    return raw_df.reset_index(drop=True).join(
        df_flags[['is_fraud','fraud_prob','flag_reason']]
    )

# ─── 4) Streamlit UI ────────────────────────────────────────────────────
st.title("🚀 VR Fraud Detector Demo")
st.write("Upload CSV; get fraud flags, human-readable reasons, and charts")

uploaded = st.file_uploader("Choose a CSV file", type="csv")
if uploaded is not None:
    raw     = pd.read_csv(uploaded)
    st.write("### Raw preview", raw.head())

    X       = preprocess(raw)
    st.write("### Engineered preview", X.head())

    results = annotate_reasons(X, raw)
    st.write("### Annotated results", results.head())

    # Download
    csv = results.to_csv(index=False)
    st.download_button("Download annotated CSV", csv,
                       file_name="vr_fraud_with_reasons.csv",
                       mime="text/csv")

    # ─── Visual summary ────────────────────────────────────────────
    st.write("## Visual summary")

    # a) Flagged vs Non-flagged
    label_map = {False:'Normal', True:'Flagged'}
    counts = results['is_fraud'].map(label_map).value_counts()
    st.write("### Transaction counts")
    st.bar_chart(counts)

    # b) Top 5 flag reasons (highest → lowest)
    reason_counts = results['flag_reason'].value_counts()
    top5 = reason_counts.nlargest(5)
    other = reason_counts.iloc[5:].sum()
    reason_summary = pd.concat([top5, pd.Series({'Other': other})])
    reason_summary = reason_summary.sort_values(ascending=False)
    st.write("### Top 5 flag reasons")
    st.bar_chart(reason_summary)

    # c) Summary metrics
    total    = len(results)
    flagged  = counts.get('Flagged', 0)
    flag_pct = round(100 * flagged / total, 1)
    avg_prob = round(results['fraud_prob'].mean(), 3)
    summary_df = pd.DataFrame({
        "Metric": ["Total txns","Flagged txns","Flag rate (%)","Avg fraud_prob"],
        "Value":  [ total, flagged, flag_pct, avg_prob ]
    })
    st.write("### Summary metrics")
    st.table(summary_df)
