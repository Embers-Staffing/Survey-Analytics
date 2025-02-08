import streamlit as st
import hmac
import os

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            # Try to get password from secrets or environment variable
            correct_password = st.secrets.get("password") or os.environ.get("STREAMLIT_PASSWORD", "9aba7500")
            
            if st.session_state["password"] == correct_password:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False
                st.error("😕 Password incorrect")
        except Exception as e:
            st.error(f"Authentication error. Please try again.")
            return False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("""
        <style>
        /* Hide Streamlit elements */
        #MainMenu, header, footer {display: none;}
        .stDeployButton {display: none;}
        .stTextInput > div > div > input {border: none !important;}
        div[data-testid="stToolbar"] {display: none;}
        div[data-testid="stDecoration"] {display: none;}
        div[data-testid="stStatusWidget"] {display: none;}
        
        /* Main styles */
        .main-title {
            font-size: 2.5em;
            font-weight: 800;
            color: #1E67C7;
            text-align: center;
            margin: 2rem 0;
            font-family: sans-serif;
        }
        
        .login-container {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .login-title {
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
            color: #1E67C7;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .custom-input {
            background: #f5f5f5;
            padding: 0.8rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        
        .login-button {
            background: #1E67C7;
            color: white;
            padding: 0.8rem;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .login-button:hover {
            background: #1850a0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 1rem;
            color: #666;
            font-size: 0.9em;
            background: #f8f9fa;
            border-top: 1px solid #eee;
        }
        
        a {
            color: #1E67C7;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main content
        st.markdown('<h1 class="main-title">Construction Career Survey Dashboard</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="login-container">
            <div class="login-title">🔒 Enter Dashboard Password</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Password input and button
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.text_input("", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered, use_container_width=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            © Embers Staffing 2025 | Built by <a href="https://github.com/ArsCodeAmatoria" target="_blank">ArsCodeAmatoria</a>
        </div>
        """, unsafe_allow_html=True)
        
        return False
    
    return st.session_state["password_correct"] 