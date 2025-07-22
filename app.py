import streamlit as st
import pandas as pd
import pickle

# ─── 1) Load the trained RF model ────────────────────────────────────────
@st.cache_data
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Immediately after loading, grab the exact feature list
FEATURE_NAMES = list(model.feature_names_in_)

# ─── 2) Preprocessing: mirror your Colab engineering ────────────────────
def preprocess(raw_df):
    df = raw_df.copy()

    # (a) Simulate market_value if missing
    if 'market_value' not in df.columns:
        df['market_value'] = df['price_paid'] * 1.0

    # (b) Price difference & overpriced flag
    df['price_difference'] = df['price_paid'] - df['market_value']
    df['is_overpriced']    = (df['price_difference'] / df['market_value']) > 0.3

    # (c) Transaction counts & repeat-user flag
    df['user_transaction_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['is_repeating_user']      = df['user_transaction_count'] > 1

    # (d) Withdrawal & suspicious-withdrawal flags
    df['is_withdrawal']         = df['type'].isin(['CASH_OUT', 'WITHDRAW'])
    df['suspicious_withdrawal'] = df['is_withdrawal'] & df['is_overpriced']

    # (e) Drop unused identifiers
    df = df.drop(columns=['user_id', 'nameDest'], errors='ignore')

    # (f) One-hot encode the `type` column
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # (g) Ensure all model features exist; fill missing with zero
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    # (h) Reorder to exactly match training
    return df[FEATURE_NAMES]

# ─── 3) Annotate reasons for human-readable flags ────────────────────────
def annotate_reasons(X, raw_df):
    # 1. Predict
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1].round(3)

    # 2. Build a small DF to hold flags + probs
    df_flags = X.copy()
    df_flags['is_fraud']   = y_pred
    df_flags['fraud_prob'] = y_proba

    # 3. Rule-based reasons
    reasons = []
    for _, row in df_flags.iterrows():
        if row['suspicious_withdrawal']:
            reasons.append("Overpriced purchase + withdrawal")
        elif row['is_overpriced']:
            reasons.append("Price > market by 30%")
        elif row['is_repeating_user']:
            reasons.append(f"Repeat user ({int(row['user_transaction_count'])} txns)")
        else:
            reasons.append("No rule triggered")
    df_flags['flag_reason'] = reasons

    # 4. Join back onto original raw DataFrame so you see all original columns
    annotated = raw_df.reset_index(drop=True).join(
        df_flags[['is_fraud','fraud_prob','flag_reason']]
    )
    return annotated

# ─── 4) Streamlit UI ────────────────────────────────────────────────────
st.title("🚀 VR Fraud Detector Demo")
st.write("Upload your VR transactions CSV; get back fraud flags, reasons, and charts.")

uploaded = st.file_uploader("Choose a CSV file", type="csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.write("### Raw data preview", raw.head())

    # 1) Engineer features
    X = preprocess(raw)
    st.write("### Engineered features preview", X.head())

    # 2) Annotate & predict
    results = annotate_reasons(X, raw)
    st.write("### Results with reasons", results.head())

    # 3) Download annotated CSV
    csv = results.to_csv(index=False)
    st.download_button(
        "Download annotated CSV",
        csv,
        file_name="vr_fraud_with_reasons.csv",
        mime="text/csv"
    )

    # 4) Interactive charts
    st.write("## Visual summary")
    st.write("### Flagged vs. Non-flagged")
    st.bar_chart(
        results['is_fraud']
               .map({0:'Normal',1:'Flagged'})
               .value_counts()
    )

    st.write("### Breakdown by reason")
    st.bar_chart(results['flag_reason'].value_counts())

    st.write("### Fraud probability distribution")
    st.line_chart(
        results['fraud_prob']
               .value_counts()
               .sort_index()
    )
