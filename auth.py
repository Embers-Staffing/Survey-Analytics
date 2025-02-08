import streamlit as st
import hmac

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False
            st.error("😕 Password incorrect")

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("""
        <style>
        .password-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .password-title {
            text-align: center;
            color: #1E67C7;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="password-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="password-title">🔒 Password Protected</h2>', unsafe_allow_html=True)
            st.text_input(
                "Please enter the password to access the dashboard",
                type="password",
                key="password"
            )
            st.button("Login", on_click=password_entered, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        return False
    
    return st.session_state["password_correct"] 