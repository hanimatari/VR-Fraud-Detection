import streamlit as st
import pandas as pd
import pickle

@st.cache(allow_output_mutation=True)
def load_model():
    with open('rf_model.pkl','rb') as f:
        return pickle.load(f)

model = load_model()

st.title("🚀 VR Fraud Detector Demo")
st.write("""
Upload a CSV of transactions (with exactly the columns your model expects),
and see which ones are flagged as fraud.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df['is_fraud']   = preds
    df['fraud_prob'] = probs.round(3)
    st.write("### Results", df)

    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False),
        file_name='vr_fraud_results.csv',
        mime='text/csv'
    )
