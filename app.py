import streamlit as st
import pandas as pd
import pickle
import yaml
import streamlit_authenticator as stauth

# ─── Auth Setup ────────────────────────────────────────────────────────────
with open('config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config.get('preauthorized', {})
)

name, auth_status, username = authenticator.login('Login', 'main')
if auth_status != True:
    if auth_status is None:
        st.warning("Please enter your username and password")
    else:
        st.error("Username/password incorrect")
    st.stop()

# ─── Model Loading ─────────────────────────────────────────────────────────
@st.cache_data
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model    = load_model()
features = list(model.feature_names_in_)

# ─── Preprocess & Annotate ────────────────────────────────────────────────
def preprocess(df):
    if 'market_value' not in df:
        df['market_value'] = df['price_paid'] * 1.0
    df['price_difference']       = df['price_paid'] - df['market_value']
    df['is_overpriced']          = (df['price_difference'] / df['market_value']) > 0.3
    df['user_transaction_count'] = df.groupby('user_id')['price_paid'].transform('count')
    df['is_repeating_user']      = df['user_transaction_count'] > 1
    df['is_withdrawal']          = df['type'].isin(['CASH_OUT','WITHDRAW'])
    df['suspicious_withdrawal']  = df['is_withdrawal'] & df['is_overpriced']
    df = df.drop(['user_id','nameDest'], axis=1, errors='ignore')
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    for c in features:
        if c not in df:
            df[c] = 0
    return df[features]

def annotate(raw):
    X      = preprocess(raw.copy())
    pred   = model.predict(X)
    prob   = model.predict_proba(X)[:,1].round(3)
    thresh = X['price_paid'].quantile(0.95)

    reasons = []
    for i, r in X.iterrows():
        if pred[i] == 0:
            reasons.append("Not suspicious")
        elif r['suspicious_withdrawal']:
            reasons.append("Overpriced + withdrawal")
        elif r['is_overpriced']:
            reasons.append("Price > market by 30%")
        elif r['is_repeating_user']:
            reasons.append(f"Repeat user ({int(r['user_transaction_count'])} txns)")
        elif r['is_withdrawal'] and r['price_paid'] > thresh:
            reasons.append(f"Large cash-out (>p95: {int(thresh)})")
        else:
            reasons.append("Other model signal")

    out = raw.reset_index(drop=True)
    out['is_fraud']         = pred
    out['fraud_prob']       = prob
    out['flag_reason']      = reasons
    out['market_value']     = X['market_value'].values
    out['price_difference'] = X['price_difference'].values
    return out

# ─── App Layout ────────────────────────────────────────────────────────────
st.sidebar.title(f"👋 Welcome, {name}")
if authenticator.logout("Sign out", "sidebar"):
    st.experimental_rerun()

st.title("🏠 VR Fraud Detector Dashboard")
st.write("Upload your transactions CSV to flag suspicious activity.")

# --- File upload
uploaded = st.file_uploader("Upload CSV", type="csv")
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
st.write("### Raw data preview", df.head())

# --- Feature preview
X_feat = preprocess(df.copy())
st.write("### Engineered features preview", X_feat.head())

# --- Annotate
res = annotate(df)
st.write("### Annotated results preview", res.head())

# --- Download
csv = res.to_csv(index=False)
st.download_button("⬇️ Download annotated CSV", csv,
                   "vr_fraud_with_reasons.csv","text/csv")

# --- Summary calculations
counts  = res['is_fraud'].value_counts().rename({0:'Normal',1:'Flagged'})
rc      = res['flag_reason'].value_counts()
top5    = rc.nlargest(5).copy()
other   = rc.iloc[5:].sum()
if other > 0:
    top5['Other'] = other

total   = len(res)
flagged = int(counts.get('Flagged', 0))
rate    = round(100 * flagged / total, 1) if total else 0

# ─── Dashboard Metrics ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total txns",     total)
col2.metric("Flagged txns",   flagged)
col3.metric("Flag rate (%)", f"{rate}%")

st.subheader("Top 5 Flag Reasons")
st.bar_chart(top5)

st.subheader("Transaction Counts")
st.bar_chart(counts)

with st.expander("📋 View full annotated table"):
    st.dataframe(res)
