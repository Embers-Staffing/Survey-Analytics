import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
from auth import check_password

# Initialize Firebase
if not firebase_admin._apps:
    # Get Firebase credentials from secrets
    firebase_config = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"],
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    }
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Set page config
st.set_page_config(
    page_title="Survey Dashboard",
    page_icon="📊",
    layout="wide"
)

# Check password
if check_password():
    # Main dashboard code
    st.title("Survey Response Dashboard")
    
    # Fetch data from Firebase
    def get_survey_data():
        responses_ref = db.collection('responses')
        docs = responses_ref.stream()
        
        data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            doc_dict['id'] = doc.id
            data.append(doc_dict)
        
        return pd.DataFrame(data)

    # Load data
    df = get_survey_data()

    # Display basic statistics
    st.header("Response Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Responses", len(df))
    
    # Add more visualizations based on your survey structure
    # Example:
    st.header("Response Analysis")
    
    # Sample visualization (modify based on your actual data structure)
    if 'category' in df.columns:
        fig = px.pie(df, names='category', title='Responses by Category')
        st.plotly_chart(fig) 