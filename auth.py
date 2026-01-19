import streamlit as st
import os
import hashlib

def load_users():
    users = {}
    raw = os.getenv("AUTH_USERS", "")
    for pair in raw.split(","):
        if ":" in pair:
            user, pwd = pair.split(":")
            users[user] = hashlib.sha256(pwd.encode()).hexdigest()
    return users

USERS = load_users()

# 🔧 LOCAL FALLBACK (ONLY if ENV is missing)
# st.warning("⚠️ AUTH_USERS environment variable not set.")


def check_password(username, password):
    if username in USERS:
        return hashlib.sha256(password.encode()).hexdigest() == USERS[username]
    return False

def login_ui():
    st.markdown("## 🔐 Login Required")
    st.markdown("Please log in to access **Health & Glow Store Location Analysis**")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_password(username, password):
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

def logout_button():
    with st.sidebar:
        st.markdown("---")
        st.write(f"👤 Logged in as **{st.session_state.get('user')}**")
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()
