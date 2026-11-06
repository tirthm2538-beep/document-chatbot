import os
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")

if "sb" not in st.session_state:
    st.session_state.sb = create_client(URL, KEY)
supabase = st.session_state.sb

if "user" not in st.session_state:
    st.session_state.user = None

st.title("Login test")

if st.session_state.user:
    st.success(f"Logged in as {st.session_state.user.email}")
    if st.button("Log out"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()
else:
    mode = st.radio("Choose", ["Sign up", "Log in"], horizontal=True)
    email = st.text_input("Email")
    password = st.text_input("Password (at least 6 characters)", type="password")
    if st.button(mode):
        try:
            if mode == "Sign up":
                res = supabase.auth.sign_up({"email": email, "password": password})
            else:
                res = supabase.auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
            if res.user and res.session:
                st.session_state.user = res.user
                st.rerun()
            else:
                st.info("Account created. Check your email to confirm it, then log in.")
        except Exception as e:
            st.error(str(e)[:200])