import streamlit as st
import pandas as pd
import pickle

# ─── 1) Load model & get feature list ───────────────────────────────────
@st.cache_data
def load_model():
    with open('rf_model.pkl','rb') as f:
        return pickle.load(f)
model = load_model()
FEATURE_NAMES = list(model.feature_names_in_)

# ─── 2) Preprocessing ───────────────────────────────────────────────────
def preprocess(raw_df):
    df = raw_df.copy()
    if 'market_value' not in df:
        df['market_value'] = df['price_paid'] * 1.0
    df['price_difference']       = df['price_paid'] - df['market_value']
    df['is_overpriced']          = (df['price_difference'] / df['market_value']) > 0.3
    df['user_transaction_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['is_repeating_user']      = df['user_transaction_count'] > 1
    df['is_withdrawal']          = df['type'].isin(['CASH_OUT','WITHDRAW'])
    df['suspicious_withdrawal']  = df['is_withdrawal'] & df['is_overpriced']
    df = df.drop(columns=['user_id','nameDest'], errors='ignore')
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    for c in FEATURE_NAMES:
        if c not in df:
            df[c] = 0
    return df[FEATURE_NAMES]

# ─── 3) Annotate reasons ─────────────────────────────────────────────────
def annotate_reasons(X, raw_df):
    # 1) Predict
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:,1].round(3)

    # 2) Determine large cash-out threshold
    large_threshold = X['price_paid'].quantile(0.95)

    # 3) Build reasons list
    reasons = []
    for _, row in X.iterrows():
        if not model.predict(row.values.reshape(1,-1))[0]:
            reasons.append("")  # not flagged
        elif row['suspicious_withdrawal']:
            reasons.append("Overpriced + withdrawal")
        elif row['is_overpriced']:
            reasons.append("Price > market by 30%")
        elif row['is_repeating_user']:
            reasons.append(f"Repeat user ({int(row['user_transaction_count'])} txns)")
        elif row['is_withdrawal'] and row['price_paid'] > large_threshold:
            reasons.append(f"Large cash‐out (>p95: {int(large_threshold)})")
        else:
            reasons.append("Other model signal")

    # 4) Assign back onto a copy of raw_df
    annotated = raw_df.copy().reset_index(drop=True)
    annotated['is_fraud']         = y_pred
    annotated['fraud_prob']       = y_proba
    annotated['flag_reason']      = reasons
    annotated['market_value']     = X['market_value'].values
    annotated['price_difference'] = X['price_difference'].values

    return annotated

# ─── 4) Streamlit UI ────────────────────────────────────────────────────
st.title("🚀 VR Fraud Detector Demo")
st.write("Upload CSV; get fraud flags, human-readable reasons, and summary charts.")

uploaded = st.file_uploader("Choose a CSV file", type="csv")
if uploaded:
    raw  = pd.read_csv(uploaded)
    st.write("### Raw preview", raw.head())

    X    = preprocess(raw)
    keep = [
        'market_value','price_difference',
        'user_transaction_count','is_overpriced',
        'is_repeating_user','is_withdrawal','suspicious_withdrawal'
    ]
    st.write("### Engineered preview", X[keep].head())

    results = annotate_reasons(X, raw)
    st.write("### Annotated results", results.head())

    csv = results.to_csv(index=False)
    st.download_button("⬇️ Download annotated CSV", csv,
                       file_name="vr_fraud_with_reasons.csv",
                       mime="text/csv")

    # ─── Visual summary ────────────────────────────────────────────
    st.write("## Visual summary")

    # 1) True counts for 0/1
    raw_counts = results['is_fraud'].value_counts().sort_index()
    normal_cnt = int(raw_counts.get(0, 0))
    flagged_cnt = int(raw_counts.get(1, 0))
    counts_named = pd.Series({'Normal': normal_cnt, 'Flagged': flagged_cnt})
    st.write("### Transaction counts")
    st.bar_chart(counts_named)

    # 2) Top 5 flag reasons + Other
    rc = results['flag_reason'].value_counts()
    top5 = rc.nlargest(5)
    other = rc.iloc[5:].sum()
    reason_summary = pd.concat([top5, pd.Series({'Other': other})])
    reason_summary = reason_summary.sort_values(ascending=False)
    st.write("### Top 5 flag reasons")
    st.bar_chart(reason_summary)

    # 3) Summary metrics table
    total = normal_cnt + flagged_cnt
    flag_rate = round(100 * flagged_cnt / total, 1) if total>0 else 0
    avg_prob = round(results['fraud_prob'].mean(), 3)
    summary_df = pd.DataFrame({
        "Metric":       ["Total txns","Flagged txns","Flag rate (%)","Avg fraud_prob"],
        "Value":        [ total,     flagged_cnt,    flag_rate,      avg_prob   ]
    })
    st.write("### Summary metrics")
    st.table(summary_df)
