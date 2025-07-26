import streamlit as st
import pandas as pd
import pickle
import openai
import traceback

# ─── DEBUG: confirm client & key ──────────────────────────────────────────
import openai, streamlit as st  # make sure these are at top of file
st.write("⚙️ openai version:", openai.__version__)
st.write("🔑 API key loaded:", bool(openai.api_key))


# ─── Config ────────────────────────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

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

# ─── Streamlit App ────────────────────────────────────────────────────────
st.title("VR Fraud Detector")
st.write("Upload your transactions CSV, get flags + an AI‐generated summary.")

up = st.file_uploader("Upload CSV", type="csv")
if not up:
    st.stop()

df = pd.read_csv(up)
st.write("### Raw data preview", df.head())

# Features & results
X_feat = preprocess(df.copy())
st.write("### Engineered features preview", X_feat.head())

res = annotate(df)
st.write("### Annotated results preview", res.head())

# Download
csv = res.to_csv(index=False)
st.download_button("⬇️ Download annotated CSV", csv,
                   "vr_fraud_with_reasons.csv","text/csv")

# Summary charts
counts = res['is_fraud'].value_counts().rename({0:'Normal',1:'Flagged'})
st.write("### Transaction counts")
st.bar_chart(counts)

rc   = res['flag_reason'].value_counts()
top5 = rc.nlargest(5).copy()
other = rc.iloc[5:].sum()
if other>0: top5['Other'] = other
st.write("### Top 5 flag reasons")
st.bar_chart(top5)

total  = len(res)
flagged= int(counts.get('Flagged',0))
rate   = round(100*flagged/total,1) if total else 0
avgp   = round(res['fraud_prob'].mean(),3)
summary = pd.DataFrame({
    'Metric':['Total txns','Flagged txns','Flag rate (%)','Avg fraud_prob'],
    'Value' :[total, flagged, rate, avgp]
})
st.write("### Summary metrics")
st.table(summary)

# ─── AI‐Generated Narrative via Chat API ────────────────────────────────
st.write("## AI Insight Summary")

prompt = (
    f"I have {total} virtual-reality asset transactions. "
    f"{flagged} were flagged as potentially fraudulent. "
    f"The top reasons are: {', '.join(top5.index.tolist())}. "
    "Write a concise 3-sentence summary for a risk officer."
)

# (optional) show the prompt
st.text("🔍 Prompt:")
st.code(prompt, language="")

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    summary_text = response.choices[0].message.content.strip()
    st.markdown(f"> {summary_text}")

except Exception as e:
    st.error(f"⚠️ OpenAI call failed: {type(e).__name__}: {e}")
    import traceback; st.text(traceback.format_exc())
