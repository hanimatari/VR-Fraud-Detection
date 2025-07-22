import streamlit as st
import pandas as pd
import pickle

@st.cache_data
def load_model():
    with open('rf_model.pkl','rb') as f:
        return pickle.load(f)
model = load_model()
# tell us exactly which columns the RF expects
FEATURE_NAMES = list(model.feature_names_in_)

FEATURE_NAMES = list(model.feature_names_in_)

def preprocess(raw_df):
    # … your existing feature engineering & dummy-alignment …
    return df[FEATURE_NAMES]

def annotate_reasons(X, raw_df):
    # … code from step 1 …
    return annotated_df

st.title("🚀 VR Fraud Detector Demo")
st.write("Upload your VR dataset and get fraud flags + reasons.")

uploaded = st.file_uploader("Choose a CSV file", type="csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.write("### Raw preview", raw.head())

    X = preprocess(raw)
    st.write("### Engineered features preview", X.head())

    results = annotate_reasons(X, raw)
    st.write("### Results with reasons", results.head())

    csv = results.to_csv(index=False)
    st.download_button("Download annotated CSV", csv,
                       file_name="vr_fraud_with_reasons.csv",
                       mime="text/csv")

    st.write("## Visual summary")
    st.write("### Flagged vs. Non-flagged")
    st.bar_chart(results['is_fraud'].value_counts().rename({0:'Normal',1:'Flagged'}))

    st.write("### Flag reasons")
    st.bar_chart(results['flag_reason'].value_counts())

    st.write("### Fraud probability distribution")
    st.line_chart(results['fraud_prob'].value_counts().sort_index())
