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
        .main-title {
            font-size: 3em;
            font-weight: 700;
            color: #1E67C7;
            text-align: center;
            padding: 40px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .password-container {
            max-width: 500px;
            margin: 40px auto;
            padding: 40px;
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        .password-title {
            text-align: center;
            color: #1E67C7;
            margin-bottom: 30px;
            font-size: 1.5em;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #6c757d;
            text-align: center;
            padding: 20px;
            font-size: 14px;
            border-top: 1px solid #dee2e6;
        }
        .stButton button {
            background-color: #1E67C7;
            color: white;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1850a0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main title
        st.markdown('<h1 class="main-title">Construction Career Survey Dashboard</h1>', unsafe_allow_html=True)
        
        # Login container
        with st.container():
            st.markdown('<div class="password-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="password-title">🔒 Password Protected</h2>', unsafe_allow_html=True)
            st.text_input(
                "Please enter the password to access the dashboard",
                type="password",
                key="password",
                placeholder="Enter password"
            )
            st.button("Login", on_click=password_entered, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Footer
        st.markdown("""
        <div class="footer">
            <p>© Embers Staffing 2025 | Built by <a href="https://github.com/ArsCodeAmatoria" target="_blank">ArsCodeAmatoria</a></p>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    return st.session_state["password_correct"] 