import streamlit as st
import pandas as pd
import pickle

# ─── 1) Load your model ────────────────────────────────────────────────
@st.cache_data
def load_model():
    with open('rf_model.pkl','rb') as f:
        return pickle.load(f)

model = load_model()
FEATURE_NAMES = list(model.feature_names_in_)   # this is the exact list of columns your RF expects

# ─── 2) A preprocessing function ───────────────────────────────────────
def preprocess(raw_df):
    df = raw_df.copy()

    # 1) (Re)simulate market_value if missing
    if 'market_value' not in df.columns:
        df['market_value'] = df['price_paid'] * 1.0

    # 2) Price difference & overpriced flag
    df['price_difference'] = df['price_paid'] - df['market_value']
    df['is_overpriced']    = (df['price_difference'] / df['market_value']) > 0.3

    # 3) Transaction counts & repeat flag
    df['user_transaction_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['is_repeating_user']      = df['user_transaction_count'] > 1

    # 4) Withdrawal flags
    df['is_withdrawal']         = df['type'].isin(['CASH_OUT','WITHDRAW'])
    df['suspicious_withdrawal'] = df['is_withdrawal'] & df['is_overpriced']

    # 5) Drop any unused columns
    df = df.drop(columns=['user_id','nameDest'], errors='ignore')

    # 6) One-hot encode `type`, drop_first=True to mirror your notebook
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # 7) Make sure every model feature exists in df; fill missing with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    # 8) Reorder columns exactly as the model expects
    return df[FEATURE_NAMES]

# ─── 3) Streamlit UI ────────────────────────────────────────────────────
st.title("🚀 VR Fraud Detector Demo")
st.write("Upload your raw VR_FRAUD_DATASET.csv and get back fraud flags & probabilities.")

uploaded = st.file_uploader("Choose a CSV file", type="csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.write("### Raw preview", raw.head())

    # ✂️  Run our preprocessing:
    X = preprocess(raw)
    st.write("### Engineered features preview", X.head())

    # ✂️  Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]

    # ✂️  Show results
    res = raw.copy()
    res['is_fraud']   = preds
    res['fraud_prob'] = probs.round(3)
    st.write("### Final results", res)

    # ✂️  Download
    st.download_button(
        "Download CSV",
        res.to_csv(index=False),
        file_name="vr_fraud_results.csv",
        mime="text/csv"
    )

