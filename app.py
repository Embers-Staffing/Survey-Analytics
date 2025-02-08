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
    try:
        # Debug print
        st.write("DataFrame columns:", df.columns.tolist())
        st.write("Sample personality data:", df['personalityTraits'].iloc[0] if 'personalityTraits' in df.columns else "No personality data")
        
        # Prepare data for clustering
        le = LabelEncoder()
        
        # Encode Holland codes
        holland_codes = []
        for idx, row in df.iterrows():
            traits = row.get('personalityTraits', {})
            if isinstance(traits, dict):
                holland = traits.get('hollandCode', [])
                if isinstance(holland, list) and holland:
                    holland_codes.append(holland[0])
                else:
                    holland_codes.append('None')
            else:
                holland_codes.append('None')
        
        # Debug print
        st.write("Holland codes:", holland_codes[:5])
        
        holland_encoded = pd.get_dummies(holland_codes)
        
        # Similar process for MBTI
        mbti_codes = []
        for idx, row in df.iterrows():
            traits = row.get('personalityTraits', {})
            if isinstance(traits, dict):
                mbti = traits.get('myersBriggs', {})
                if isinstance(mbti, dict):
                    attention = mbti.get('attention', [])
                    if isinstance(attention, list) and attention:
                        mbti_codes.append(attention[0])
                    else:
                        mbti_codes.append('None')
                else:
                    mbti_codes.append('None')
            else:
                mbti_codes.append('None')
        
        # Debug print
        st.write("MBTI codes:", mbti_codes[:5])
        
        mbti_encoded = pd.get_dummies(mbti_codes)
        
        # Combine features
        features = pd.concat([holland_encoded, mbti_encoded], axis=1)
        
        # Debug print
        st.write("Feature columns:", features.columns.tolist())
        
        if features.empty or features.shape[1] == 0:
            raise ValueError("No features available for clustering")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        return clusters
    except Exception as e:
        st.error(f"Unable to perform personality clustering: {str(e)}")
        return [0] * len(df)

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
            # Keep original nested structure
            data.append(doc_dict)
            
            # Debug print
            st.write("Sample data structure:", doc_dict.keys())
            if 'personalityTraits' in doc_dict:
                st.write("Personality structure:", doc_dict['personalityTraits'])
            if 'skills' in doc_dict:
                st.write("Skills structure:", doc_dict['skills'])
        
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
            try:
                years_mean = round(df['personal_yearsInConstruction'].astype(float).mean(), 1)
                st.metric("Average Years in Construction", years_mean)
            except:
                st.metric("Average Years in Construction", "N/A")
        with col3:
            try:
                role = df['skills_experience_role'].mode()[0] if not df['skills_experience_role'].empty else "N/A"
                st.metric("Most Common Role", role)
            except:
                st.metric("Most Common Role", "N/A")

        # Basic visualizations (using seaborn)
        try:
            st.subheader("Years in Construction Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='personal_yearsInConstruction', bins=20, ax=ax)
            st.pyplot(fig)
        except:
            st.write("No construction years data available")

        # Career Goals Analysis with enhanced styling
        st.header("Career Development")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Convert list column to individual rows
                goals_list = []
                for idx, row in df.iterrows():
                    if isinstance(row.get('careerGoals'), list):
                        goals_list.extend(row['careerGoals'])
                career_goals = pd.DataFrame(goals_list, columns=['goal'])
                fig = px.pie(career_goals, names='goal', title='Career Goals Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write("No career goals data available")
        
        with col2:
            try:
                fig = px.pie(df, names='advancementPreference', title='Advancement Preferences',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.write("No advancement preference data available")

    with tab2:
        st.header("Advanced Analytics")
        
        # Personality Clustering
        st.subheader("Personality Clusters")
        try:
            clusters = analyze_personality_clusters(df)
            df['Cluster'] = clusters
            
            fig = px.scatter(df, 
                            x='personal_yearsInConstruction', 
                            y='targetSalary',
                            color='Cluster',
                            title='Experience vs Salary Expectations by Personality Cluster')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.write("Unable to generate personality cluster visualization")
        
        # Skills Analysis
        st.subheader("Skills Distribution")
        try:
            tech_skills = []
            for idx, row in df.iterrows():
                if isinstance(row.get('skills', {}).get('technical'), list):
                    tech_skills.extend(row['skills']['technical'])
            tech_skills_df = pd.DataFrame(tech_skills, columns=['skill'])
            fig = px.histogram(tech_skills_df, x='skill', 
                             title='Technical Skills Distribution',
                             color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.write("No technical skills data available")
        
        # Personality Insights with Advanced Visualization
        st.subheader("Personality Type Distribution")
        try:
            mbti_data = []
            for idx, row in df.iterrows():
                attention = row.get('personalityTraits', {}).get('myersBriggs', {}).get('attention', [])
                if isinstance(attention, list) and attention:
                    mbti_data.append(attention[0])
            mbti_df = pd.DataFrame(mbti_data, columns=['MBTI'])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=mbti_df, x='MBTI', palette='viridis', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except:
            st.write("No MBTI data available")

    with tab3:
        st.header("Raw Data")
        st.dataframe(df) 