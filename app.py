import streamlit as st
import pandas as pd
import pickle

def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
features = list(model.feature_names_in_)

def preprocess(df):
    if 'market_value' not in df:
        df['market_value'] = df['price_paid'] * 1.0
    df['price_diff'] = df['price_paid'] - df['market_value']
    df['overpriced'] = (df['price_diff'] / df['market_value']) > 0.3
    df['trans_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['repeat_user'] = df['trans_count'] > 1
    df['withdraw'] = df['type'].isin(['CASH_OUT', 'WITHDRAW'])
    df['suspicious'] = df['withdraw'] & df['overpriced']
    df = df.drop(['user_id', 'nameDest'], axis=1, errors='ignore')
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    for c in features:
        if c not in df:
            df[c] = 0
    return df[features]

def annotate(raw):
    X = preprocess(raw.copy())
    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1].round(3)
    thresh = X['price_paid'].quantile(0.95)
    reasons = []
    for i, row in X.iterrows():
        if pred[i] == 0:
            reasons.append("Not suspicious")
        elif row['suspicious']:
            reasons.append("Overpriced + withdrawal")
        elif row['overpriced']:
            reasons.append("Price > market by 30%")
        elif row['repeat_user']:
            reasons.append(f"Repeat user ({int(row['trans_count'])} txns)")
        elif row['withdraw'] and row['price_paid'] > thresh:
            reasons.append(f"Large cash‐out (>p95: {int(thresh)})")
        else:
            reasons.append("Other model signal")
    out = raw.reset_index(drop=True)
    out['is_fraud'] = pred
    out['fraud_prob'] = prob
    out['flag_reason'] = reasons
    return out

st.title("VR Fraud Detector")
up = st.file_uploader("Upload CSV", type="csv")

if up:
    df = pd.read_csv(up)
    st.write("Raw data", df.head())

    X = preprocess(df.copy())
    st.write("Features", X.head())

    res = annotate(df)
    st.write("Results", res.head())

    st.download_button(
        "Download CSV",
        res.to_csv(index=False),
        "results.csv",
        "text/csv"
    )

    counts = res['is_fraud'].value_counts().rename({0: 'normal', 1: 'fraud'})
    st.write("### Transaction counts")
    st.bar_chart(counts)

    rc = res['flag_reason'].value_counts()
    top5 = rc.nlargest(5).copy()
    other = rc.iloc[5:].sum()
    if other > 0:
        top5['Other'] = other
    st.write("### Top 5 flag reasons")
    st.bar_chart(top5)

    total = len(res)
    flagged = int(counts.get('fraud', 0))
    rate = round(100 * flagged / total, 1) if total else 0
    avgp = round(res['fraud_prob'].mean(), 3)
    summary = pd.DataFrame({
        'Metric': ['Total txns', 'Flagged txns', 'Flag rate (%)', 'Avg fraud_prob'],
        'Value':  [total, flagged, rate, avgp]
    })
    st.write("### Summary metrics")
    st.table(summary)
