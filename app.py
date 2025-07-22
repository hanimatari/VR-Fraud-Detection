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
    df['withdraw'] = df['type'].isin(['CASH_OUT','WITHDRAW'])
    df['suspicious'] = df['withdraw'] & df['overpriced']
    df = df.drop(['user_id','nameDest'], axis=1, errors='ignore')
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    for c in features:
        if c not in df:
            df[c] = 0
    return df[features]

def annotate(raw):
    X = preprocess(raw.copy())
    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    out = raw.reset_index(drop=True)
    out['is_fraud'] = pred
    out['fraud_prob'] = prob.round(3)
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
    csv = res.to_csv(index=False)
    st.download_button("Download CSV", csv, "results.csv", "text/csv")
    counts = res['is_fraud'].value_counts().rename({0:'normal',1:'fraud'})
    st.bar_chart(counts)
