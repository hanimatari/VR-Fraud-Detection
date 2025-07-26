import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
from datetime import datetime

# ─── 0) DB SETUP ──────────────────────────────────────────────────────────
conn = sqlite3.connect("data.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
  username TEXT PRIMARY KEY,
  password_hash TEXT,
  is_admin INTEGER
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS uploads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT,
  uploaded_by TEXT,
  timestamp TEXT
)
""")
conn.commit()

# bootstrap default admin
if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
    default_pw = "admin123"
    h = hashlib.sha256(default_pw.encode()).hexdigest()
    c.execute(
      "INSERT INTO users(username,password_hash,is_admin) VALUES (?,?,1)",
      ("admin", h)
    )
    conn.commit()

# ─── 1) LOGIN FLOW ────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 VR Fraud Detector — Login")
    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")
    if st.button("Log in"):
        row = c.execute(
          "SELECT password_hash, is_admin FROM users WHERE username=?",
          (user,)
        ).fetchone()
        if row and hashlib.sha256(pwd.encode()).hexdigest() == row[0]:
            st.session_state.logged_in = True
            st.session_state.user      = user
            st.session_state.is_admin  = bool(row[1])
            st.experimental_rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ─── LOGOUT BUTTON ────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("🔒 Log out"):
        for k in ["logged_in","user","is_admin"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

# ─── 2) PAGE NAVIGATION ───────────────────────────────────────────────────
if st.session_state.is_admin:
    page = st.sidebar.selectbox("Go to", ["Admin Dashboard","Fraud Detector"])
else:
    page = "Fraud Detector"

# ─── 3) MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_data
def load_model():
    with open("model.pkl","rb") as f:
        return pickle.load(f)
model    = load_model()
features = list(model.feature_names_in_)

# ─── 4) FEATURE ENGINEERING & ANNOTATION ─────────────────────────────────
def preprocess(df):
    if "market_value" not in df:
        df["market_value"] = df["price_paid"] * 1.0
    df["price_difference"]       = df["price_paid"] - df["market_value"]
    df["is_overpriced"]          = (df["price_difference"]/df["market_value"]) > 0.3
    df["user_transaction_count"] = df.groupby("user_id")["price_paid"].transform("count")
    df["is_repeating_user"]      = df["user_transaction_count"] > 1
    df["is_withdrawal"]          = df["type"].isin(["CASH_OUT","WITHDRAW"])
    df["suspicious_withdrawal"]  = df["is_withdrawal"] & df["is_overpriced"]
    df = df.drop(["user_id","nameDest"], axis=1, errors="ignore")
    df = pd.get_dummies(df, columns=["type"], drop_first=True)
    for feat in features:
        if feat not in df:
            df[feat] = 0
    return df[features]

def annotate(raw):
    X      = preprocess(raw.copy())
    pred   = model.predict(X)
    prob   = model.predict_proba(X)[:,1].round(3)
    thresh = X["price_paid"].quantile(0.95)

    reasons = []
    for i, row in X.iterrows():
        if pred[i] == 0:
            reasons.append("Not suspicious")
        elif row["suspicious_withdrawal"]:
            reasons.append("Overpriced + withdrawal")
        elif row["is_overpriced"]:
            reasons.append("Price > market by 30%")
        elif row["is_repeating_user"]:
            reasons.append(f"Repeat user ({int(row['user_transaction_count'])} txns)")
        elif row["is_withdrawal"] and row["price_paid"] > thresh:
            reasons.append("Large cash-out (>p95)")
        else:
            reasons.append("Other signal")

    out = raw.reset_index(drop=True)
    out["is_fraud"]         = pred
    out["fraud_prob"]       = prob
    out["flag_reason"]      = reasons
    out["market_value"]     = X["market_value"].values
    out["price_difference"] = X["price_difference"].values
    return out

# ─── 5) ADMIN DASHBOARD ───────────────────────────────────────────────────
if page == "Admin Dashboard":
    st.title("⚙️ Admin Dashboard")
    st.write(f"Logged in as **{st.session_state.user}** (admin)")

    st.subheader("Create new user")
    with st.form("frm"):
        new_u = st.text_input("Username")
        new_p = st.text_input("Password", type="password")
        is_a  = st.checkbox("Make admin")
        if st.form_submit_button("➕ Add user"):
            if new_u and new_p:
                h = hashlib.sha256(new_p.encode()).hexdigest()
                try:
                    c.execute(
                      "INSERT INTO users(username,password_hash,is_admin) VALUES (?,?,?)",
                      (new_u, h, 1 if is_a else 0)
                    )
                    conn.commit()
                    st.success(f"User `{new_u}` added")
                    st.experimental_rerun()
                except sqlite3.IntegrityError:
                    st.error("Username already exists")

    st.subheader("Existing users")
    users = c.execute("SELECT username, is_admin FROM users").fetchall()
    df_u = pd.DataFrame(users, columns=["username","is_admin"])
    st.dataframe(df_u, use_container_width=True)

    st.subheader("Upload history")
    logs = c.execute(
      "SELECT filename, uploaded_by, timestamp FROM uploads ORDER BY id DESC"
    ).fetchall()
    df_l = pd.DataFrame(logs, columns=["filename","user","when"])
    st.dataframe(df_l, use_container_width=True)

# ─── 6) FRAUD DETECTOR ─────────────────────────────────────────────────────
if page == "Fraud Detector":
    st.title("🚀 VR Fraud Detector")
    st.write(f"Welcome, **{st.session_state.user}**")

    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        c.execute(
          "INSERT INTO uploads(filename,uploaded_by,timestamp) VALUES (?,?,?)",
          (up.name, st.session_state.user, datetime.utcnow().isoformat())
        )
        conn.commit()

        df = pd.read_csv(up)
        st.subheader("Raw preview")
        st.dataframe(df.head(), use_container_width=True)

        Xf = preprocess(df.copy())
        st.subheader("Features")
        st.dataframe(Xf.head(), use_container_width=True)

        res = annotate(df)
        st.subheader("Results")
        st.dataframe(res.head(), use_container_width=True)

        st.download_button(
            "⬇️ Download results.csv",
            res.to_csv(index=False),
            "results.csv",
            "text/csv"
        )

        st.subheader("Counts")
        cnts = res["is_fraud"].value_counts().rename({0:"Normal",1:"Flagged"})
        st.bar_chart(cnts)

        st.subheader("Top 5 reasons")
        rc    = res["flag_reason"].value_counts()
        top5  = rc.nlargest(5).copy()
        other = rc.iloc[5:].sum()
        if other>0:
            top5["Not suspicious"] = other
        st.bar_chart(top5)

        st.subheader("Summary metrics")
        tot   = len(res)
        flg   = int(cnts.get("Flagged",0))
        rate  = round(100*flg/tot,1)
        avg_p = round(res["fraud_prob"].mean(),3)
        df_m  = pd.DataFrame({
          "Metric": ["Total","Flagged","Flag rate (%)","Avg prob"],
          "Value" : [tot, flg, rate, avg_p]
        })
        st.table(df_m)
