import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
from datetime import datetime
from PIL import Image
import base64  # added for image encoding

# ─── 0) DB SETUP ──────────────────────────────────────────────────────────
conn = sqlite3.connect("data.db", check_same_thread=False)
c    = conn.cursor()

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

# bootstrap default admin if none exist
if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
    default_pw = "admin123"
    h = hashlib.sha256(default_pw.encode()).hexdigest()
    c.execute("INSERT INTO users VALUES (?,?,1)", ("admin", h))
    conn.commit()

# ─── 1) LOGIN FLOW ────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if not st.session_state.logged_in:
    # Display logo above
    logo_path = "cash_catchers_logo.jpeg"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=200)
    except:
        st.warning("Logo not found. Make sure 'cash_catchers_logo.jpeg' exists.")

    # Title with embedded logo instead of lock emoji
    try:
        logo_base64 = get_image_base64(logo_path)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="data:image/jpeg;base64,{logo_base64}" width="40" style="margin-bottom: 6px;">
                <h1 style="margin: 0;">VR Fraud Detector — Login</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        st.title("VR Fraud Detector — Login")

    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")
    if st.button("Log in"):
        row = c.execute(
            "SELECT password_hash,is_admin FROM users WHERE username=?",
            (user,)
        ).fetchone()
        if row and hashlib.sha256(pwd.encode()).hexdigest() == row[0]:
            st.session_state.logged_in = True
            st.session_state.user      = user
            st.session_state.is_admin  = bool(row[1])
            st.success("✅ Logged in!")
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ─── LOGOUT BUTTON ────────────────────────────────────────────────────────
if st.sidebar.button("🔒 Log out"):
    for k in ["logged_in","user","is_admin"]:
        st.session_state.pop(k, None)
    st.success("Logged out")
    st.stop()

# ─── 2) PAGE NAVIGATION ────────────────────────────────────────────────────
if st.session_state.is_admin:
    page = st.sidebar.selectbox("Go to", ["Admin Dashboard","Fraud Detector"])
else:
    page = "Fraud Detector"

# ─── 3) MODEL LOADIN
