import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
from auth import check_password

# Set style for seaborn
sns.set_theme(style="whitegrid")
plt.style.use('default')

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

def create_correlation_heatmap(df, columns):
    # Create correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Correlation Heatmap')
    return fig

def analyze_personality_clusters(df):
    # Prepare data for clustering
    le = LabelEncoder()
    
    # Encode Holland codes
    holland_encoded = pd.get_dummies(df['personalityTraits.hollandCode'].apply(lambda x: x[0] if x else 'None'))
    
    # Encode MBTI
    mbti_encoded = pd.get_dummies(df['personalityTraits.myersBriggs.attention'].apply(lambda x: x[0] if x else 'None'))
    
    # Combine features
    features = pd.concat([holland_encoded, mbti_encoded], axis=1)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    return clusters

# Check password
if check_password():
    st.title("Survey Response Dashboard")
    
    # Fetch and prepare data
    def get_survey_data():
        responses_ref = db.collection('responses')
        docs = responses_ref.stream()
        
        data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            # Flatten nested structures
            if 'personalInfo' in doc_dict:
                for key, value in doc_dict['personalInfo'].items():
                    doc_dict[f'personal_{key}'] = value
            if 'workPreferences' in doc_dict:
                for key, value in doc_dict['workPreferences'].items():
                    doc_dict[f'work_{key}'] = value
            if 'skills' in doc_dict:
                for key, value in doc_dict['skills'].items():
                    doc_dict[f'skills_{key}'] = value
            data.append(doc_dict)
        
        return pd.DataFrame(data)

    df = get_survey_data()

    # Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Advanced Analytics", "Raw Data"])

    with tab1:
        # Overview Section
        st.header("Response Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Responses", len(df))
        with col2:
            st.metric("Average Years in Construction", 
                     round(df['personal_yearsInConstruction'].astype(float).mean(), 1))
        with col3:
            st.metric("Most Common Role", 
                     df['skills.experience.role'].mode()[0] if not df['skills.experience.role'].empty else "N/A")

        # Basic visualizations (using seaborn)
        st.subheader("Years in Construction Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='personal_yearsInConstruction', bins=20, ax=ax)
        st.pyplot(fig)

        # Career Goals Analysis with enhanced styling
        st.header("Career Development")
        col1, col2 = st.columns(2)
        
        with col1:
            career_goals = pd.DataFrame([goal for goals in df['careerGoals'] for goal in goals])
            fig = px.pie(career_goals, names=0, title='Career Goals Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(df, names='advancementPreference', title='Advancement Preferences',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Advanced Analytics")
        
        # Personality Clustering
        st.subheader("Personality Clusters")
        clusters = analyze_personality_clusters(df)
        df['Cluster'] = clusters
        
        fig = px.scatter(df, 
                        x='personal_yearsInConstruction', 
                        y='targetSalary',
                        color='Cluster',
                        title='Experience vs Salary Expectations by Personality Cluster')
        st.plotly_chart(fig, use_container_width=True)
        
        # Skills Analysis
        st.subheader("Skills Distribution")
        tech_skills = pd.DataFrame([skill for skills in df['skills.technical'] for skill in skills])
        fig = px.histogram(tech_skills, x=0, 
                          title='Technical Skills Distribution',
                          color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Personality Insights with Advanced Visualization
        st.subheader("Personality Type Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='personalityTraits.myersBriggs.attention',
                     palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab3:
        st.header("Raw Data")
        st.dataframe(df) 