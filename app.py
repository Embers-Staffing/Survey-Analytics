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
                # Access nested personalInfo structure
                years = [float(row.get('personalInfo', {}).get('yearsInConstruction', 0)) 
                        for row in df.to_dict('records')]
                years_mean = round(sum(years) / len(years), 1)
                st.metric("Average Years in Construction", years_mean)
            except:
                st.metric("Average Years in Construction", "N/A")
        with col3:
            try:
                # Access nested skills structure
                roles = [row.get('skills', {}).get('experience', {}).get('role', 'Unknown') 
                        for row in df.to_dict('records')]
                most_common = max(set(roles), key=roles.count)
                st.metric("Most Common Role", most_common)
            except:
                st.metric("Most Common Role", "N/A")

        # Basic visualizations (using seaborn)
        try:
            st.subheader("Years in Construction Distribution")
            years_data = pd.DataFrame([{
                'Years': float(row.get('personalInfo', {}).get('yearsInConstruction', 0))
            } for row in df.to_dict('records')])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=years_data, x='Years', bins=20, ax=ax)
            plt.title('Years in Construction Distribution')
            plt.xlabel('Years')
            st.pyplot(fig)
        except:
            st.write("No construction years data available")

        # Career Goals Analysis with enhanced styling
        st.header("Career Development")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Extract career goals from nested structure
                goals_list = []
                for row in df.to_dict('records'):
                    goals = row.get('goals', {}).get('careerGoals', [])
                    if isinstance(goals, list):
                        goals_list.extend(goals)
                
                if goals_list:
                    career_goals = pd.DataFrame(goals_list, columns=['goal'])
                    fig = px.pie(career_goals, names='goal', title='Career Goals Distribution',
                                color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No career goals data available")
            except Exception as e:
                st.write("No career goals data available")
        
        with col2:
            try:
                # Extract advancement preferences from nested structure
                prefs = [row.get('goals', {}).get('advancementPreference', 'Unknown') 
                        for row in df.to_dict('records')]
                pref_df = pd.DataFrame(prefs, columns=['preference'])
                
                fig = px.pie(pref_df, names='preference', title='Advancement Preferences',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.write("No advancement preference data available")

    with tab2:
        st.header("Advanced Analytics")
        
        # Personality Clustering
        st.subheader("Personality Clusters")
        try:
            # Create features for clustering
            features_data = []
            for _, row in df.iterrows():
                try:
                    # Get years in construction
                    years_str = row.get('personalInfo', {}).get('yearsInConstruction', '0')
                    years = float(years_str) if years_str.isdigit() else 0.0
                    
                    # Get role and project size
                    experience = row.get('skills', {}).get('experience', {})
                    role = experience.get('role', 'Unknown')
                    project_size = experience.get('projectSize', 'Unknown')
                    
                    # Get personality traits
                    traits = row.get('personalityTraits', {})
                    mbti = traits.get('myersBriggs', {})
                    holland = traits.get('hollandCode', [])
                    
                    # Combine MBTI traits
                    personality_type = (
                        mbti.get('attention', [''])[0] + 
                        mbti.get('information', [''])[0] + 
                        mbti.get('decisions', [''])[0] + 
                        mbti.get('lifestyle', [''])[0]
                    )
                    
                    # Get career preferences
                    work_prefs = row.get('workPreferences', {})
                    environment = work_prefs.get('environment', 'Unknown')
                    travel = work_prefs.get('travelWillingness', 'Unknown')
                    
                    if years >= 0 and role and personality_type:
                        features_data.append({
                            'years': years,
                            'role': role,
                            'project_size': project_size,
                            'personality': personality_type,
                            'holland_primary': holland[0] if holland else 'Unknown',
                            'environment': environment,
                            'travel': travel
                        })
                except (ValueError, TypeError, AttributeError) as e:
                    continue
            
            if not features_data:
                raise ValueError("No valid data for clustering")
            
            # Create DataFrame for clustering
            features_df = pd.DataFrame(features_data)
            
            # Encode categorical variables
            categorical_columns = ['role', 'project_size', 'personality', 'holland_primary', 'environment', 'travel']
            encoded_features = pd.get_dummies(features_df[categorical_columns])
            
            # Combine with numerical features
            X = pd.concat([features_df[['years']], encoded_features], axis=1)
            
            # Normalize features
            X = (X - X.mean()) / X.std()
            
            # Let user select number of clusters
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=3)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            features_df['Cluster'] = clusters
            
            # Visualization options
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["3D Scatter", "2D Scatter Matrix", "Feature Importance"]
            )
            
            if viz_type == "3D Scatter":
                fig = px.scatter_3d(
                    features_df,
                    x='years',
                    y='role',
                    z='personality',
                    color='Cluster',
                    title='Experience and Personality Clusters',
                    labels={
                        'years': 'Years in Construction',
                        'role': 'Role',
                        'personality': 'Personality Type'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "2D Scatter Matrix":
                fig = px.scatter_matrix(
                    features_df,
                    dimensions=['years', 'project_size', 'environment'],
                    color='Cluster',
                    title='Feature Relationships by Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Feature Importance
                # Calculate feature importance based on cluster separation
                feature_importance = {}
                for col in X.columns:
                    variance_ratio = np.var(X[col]) / np.var(X[col].groupby(clusters).transform('mean'))
                    feature_importance[col] = variance_ratio
                
                # Plot feature importance
                fig = px.bar(
                    x=list(feature_importance.keys()),
                    y=list(feature_importance.values()),
                    title='Feature Importance in Clustering',
                    labels={'x': 'Feature', 'y': 'Importance Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show cluster summary
            st.subheader("Cluster Summary")
            for cluster in range(n_clusters):
                cluster_data = features_df[features_df['Cluster'] == cluster]
                st.write(f"Cluster {cluster + 1}:")
                st.write(f"- Size: {len(cluster_data)} members ({(len(cluster_data)/len(features_df)*100):.1f}%)")
                st.write(f"- Average Years: {cluster_data['years'].mean():.1f}")
                st.write(f"- Common Roles: {', '.join(cluster_data['role'].value_counts().head(2).index)}")
                st.write(f"- Common Project Sizes: {', '.join(cluster_data['project_size'].value_counts().head(2).index)}")
                st.write(f"- Common Personality Types: {', '.join(cluster_data['personality'].value_counts().head(2).index)}")
                st.write(f"- Preferred Environments: {', '.join(cluster_data['environment'].value_counts().head(2).index)}")
                st.write("---")
                
        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            st.write("Unable to generate personality cluster visualization")
        
        # Skills Analysis
        st.subheader("Skills Distribution")
        try:
            # Extract technical skills
            all_skills = []
            for _, row in df.iterrows():
                skills = row.get('skills', {})
                if isinstance(skills, dict):
                    tech = skills.get('technical', [])
                    if isinstance(tech, list):
                        all_skills.extend(tech)
            
            if all_skills:
                skills_df = pd.DataFrame(all_skills, columns=['Skill'])
                fig = px.histogram(
                    skills_df,
                    x='Skill',
                    title='Technical Skills Distribution',
                    color_discrete_sequence=['#2ecc71']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No technical skills found in the data")
        except Exception as e:
            st.error(f"Skills analysis error: {str(e)}")
            st.write("No technical skills data available")
        
        # Personality Insights
        st.subheader("Personality Type Distribution")
        try:
            # Extract MBTI data
            mbti_types = []
            for _, row in df.iterrows():
                traits = row.get('personalityTraits', {})
                if isinstance(traits, dict):
                    mbti = traits.get('myersBriggs', {})
                    if isinstance(mbti, dict):
                        attention = mbti.get('attention', [])
                        if isinstance(attention, list) and attention:
                            mbti_types.append(attention[0])
            
            if mbti_types:
                # Create DataFrame and get value counts
                mbti_counts = pd.Series(mbti_types).value_counts()
                
                # Create bar chart
                fig = px.bar(
                    x=mbti_counts.index,
                    y=mbti_counts.values,
                    title='MBTI Type Distribution',
                    labels={'x': 'MBTI Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary
                st.write("MBTI Type Breakdown:")
                for mbti_type, count in mbti_counts.items():
                    st.write(f"- {mbti_type}: {count} responses")
            else:
                st.write("No MBTI data found in the responses")
        except Exception as e:
            st.error(f"MBTI analysis error: {str(e)}")
            st.write("No MBTI data available")

        # Regression Analysis
        st.header("Regression Analysis")
        
        try:
            # Prepare data for regression
            regression_data = []
            for _, row in df.iterrows():
                try:
                    years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                    salary_level = {
                        'entry': 1,
                        'mid': 2,
                        'senior': 3,
                        'executive': 4
                    }.get(row.get('goals', {}).get('targetSalary', 'entry'), 1)
                    
                    project_size_map = {
                        'small': 1,
                        'medium': 2,
                        'large': 3
                    }
                    project_size = project_size_map.get(
                        row.get('skills', {}).get('experience', {}).get('projectSize', 'small'),
                        1
                    )
                    
                    regression_data.append({
                        'Years': years,
                        'Project Size': project_size,
                        'Target Salary Level': salary_level
                    })
                except:
                    continue
            
            regression_df = pd.DataFrame(regression_data)
            
            # Correlation Analysis
            st.subheader("Correlation Analysis")
            corr = regression_df.corr()
            fig = px.imshow(
                corr,
                text=corr.round(2),
                aspect='auto',
                title='Correlation Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter Plots with Trend Lines
            st.subheader("Relationship Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Years vs Salary Level
                fig = px.scatter(
                    regression_df,
                    x='Years',
                    y='Target Salary Level',
                    trendline="ols",
                    title='Years of Experience vs Target Salary Level',
                    labels={
                        'Target Salary Level': 'Salary Level (1=Entry, 4=Executive)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation coefficient
                correlation = regression_df['Years'].corr(regression_df['Target Salary Level'])
                st.write(f"Correlation coefficient: {correlation:.2f}")
            
            with col2:
                # Project Size vs Salary Level
                fig = px.scatter(
                    regression_df,
                    x='Project Size',
                    y='Target Salary Level',
                    trendline="ols",
                    title='Project Size vs Target Salary Level',
                    labels={
                        'Project Size': 'Project Size (1=Small, 3=Large)',
                        'Target Salary Level': 'Salary Level (1=Entry, 4=Executive)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation coefficient
                correlation = regression_df['Project Size'].corr(regression_df['Target Salary Level'])
                st.write(f"Correlation coefficient: {correlation:.2f}")
            
            # Additional Analysis
            st.subheader("Career Progression Analysis")
            
            # Box plot of years by target salary level
            fig = px.box(
                regression_df,
                x='Target Salary Level',
                y='Years',
                title='Years of Experience Distribution by Target Salary Level',
                labels={
                    'Target Salary Level': 'Salary Level (1=Entry, 4=Executive)',
                    'Years': 'Years of Experience'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.write("Summary Statistics by Target Salary Level:")
            summary = regression_df.groupby('Target Salary Level').agg({
                'Years': ['mean', 'std', 'count'],
                'Project Size': 'mean'
            }).round(2)
            st.dataframe(summary)
            
        except Exception as e:
            st.error(f"Regression analysis error: {str(e)}")
            st.write("Unable to perform regression analysis")

    with tab3:
        st.header("Raw Data")
        st.dataframe(df) 