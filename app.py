# Standard imports
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

# Constants
HOLLAND_CODES = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']

# Set style for seaborn
sns.set_theme(style="whitegrid")
plt.style.use('default')

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["firebase"])
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Page config
st.set_page_config(
    page_title="Construction Career Survey Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add Survey Button with improved UI
st.markdown("""
<style>
.survey-button {
    position: fixed;
    right: 30px;
    top: 75px;
    background-color: #1E67C7;  /* Professional blue */
    color: white !important;  /* Force white text */
    padding: 12px 24px;
    text-decoration: none !important;
    border-radius: 25px;
    font-weight: 700;
    font-size: 16px;
    z-index: 999;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    border: 2px solid #1E67C7;
    opacity: 1 !important;  /* Ensure full opacity */
}
.survey-button:hover {
    background-color: #ffffff;
    color: #1E67C7 !important;  /* Force blue text on hover */
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
    text-decoration: none !important;
    opacity: 1 !important;
}
</style>
<a href="https://career-survey.netlify.app" target="_blank" class="survey-button">TAKE THE SURVEY</a>
""", unsafe_allow_html=True)

# Add section headers with icons and descriptions
def section_header(icon, title, description):
    st.markdown(f"""
    <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
        <h2 style='margin:0;'>{icon} {title}</h2>
        <p style='color: #6C757D; margin-top: 0.5rem;'>{description}</p>
    </div>
    """, unsafe_allow_html=True)

# Add loading spinners and progress bars
with st.spinner("Loading data..."):
    # Your existing data loading code...
    pass

# Add tooltips and help text
st.sidebar.info("Use these filters to customize the dashboard view")

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

def get_survey_data():
    """Fetch and process survey responses from Firebase."""
    try:
        responses_ref = db.collection('responses')
        docs = responses_ref.stream()
        data = []
        count = 0
        latest_date = None
        
        for doc in docs:
            doc_data = doc.to_dict()
            
            # Convert submittedAt to UTC timezone
            if 'submittedAt' in doc_data:
                try:
                    # Parse the ISO format date and localize to UTC
                    submission_date = pd.to_datetime(doc_data['submittedAt']).tz_localize('UTC')
                    doc_data['submittedAt'] = submission_date
                    
                    # Track latest submission
                    if latest_date is None or submission_date > latest_date:
                        latest_date = submission_date
                except:
                    # If date parsing fails, skip updating the date
                    pass
            
            data.append(doc_data)
            count += 1
        
        # Show stats in sidebar
        st.sidebar.markdown("### Database Stats")
        st.sidebar.markdown(f"**Total Responses:** {count}")
        if latest_date:
            st.sidebar.markdown(f"**Latest Submission:** {latest_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        st.sidebar.markdown(f"**Last Checked:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def create_sidebar_filters(df: pd.DataFrame) -> dict:
    """Create sidebar filters for the dashboard."""
    st.sidebar.header("Filters")
    
    # Role Filter
    roles = set()
    for _, row in df.iterrows():
        if isinstance(row.get('skills'), dict):
            experience = row['skills'].get('experience', {})
            if isinstance(experience, dict):
                role = experience.get('role')
                if role:
                    roles.add(role)
    selected_roles = st.sidebar.multiselect("Roles", list(roles) if roles else [])
    
    # Experience Level Filter
    years_ranges = ["0-2 years", "3-5 years", "5-10 years", "10+ years"]
    selected_exp = st.sidebar.multiselect("Experience Level", years_ranges)
    
    # Date Range Filter with better validation
    if 'submittedAt' in df.columns:
        try:
            # Convert to datetime with error handling
            dates = pd.to_datetime(df['submittedAt'], errors='coerce', utc=True)
            valid_dates = dates.dropna()
            
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                
                # Show available date range
                st.sidebar.info(f"Survey responses from: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                
                # Default to most recent 30 days of data
                default_start = max(min_date, (max_date - timedelta(days=30)))
                
                # Ensure selected dates are within valid range
                selected_dates = st.sidebar.date_input(
                    "Select Date Range",
                    value=(default_start, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_input"
                )
                
                # Validate selected dates
                if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    # Ensure dates are within valid range
                    start_date = max(min_date, start_date)
                    end_date = min(max_date, end_date)
                else:
                    # If single date selected, use it for both start and end
                    start_date = end_date = selected_dates
                    start_date = max(min_date, min(max_date, start_date))
                    end_date = start_date
                
                return {
                    'date_range': (start_date, end_date),
                    'roles': selected_roles,
                    'experience': selected_exp
                }
            else:
                st.sidebar.warning("No valid survey dates found")
        except Exception as e:
            st.sidebar.error(f"Date filter error: {str(e)}")
    
    return {
        'date_range': None,
        'roles': selected_roles,
        'experience': selected_exp
    }

def apply_filters(df, filters):
    """Apply selected filters to the dataframe."""
    filtered_df = df.copy()
    
    if filters.get('date_range'):
        try:
            dates = pd.to_datetime(filtered_df['submittedAt'], errors='coerce', utc=True)
            mask = (dates.dt.date >= filters['date_range'][0]) & \
                   (dates.dt.date <= filters['date_range'][1])
            filtered_df = filtered_df[mask]
        except Exception as e:
            st.warning(f"Error applying date filter: {str(e)}")
    
    if filters.get('roles'):
        filtered_df = filtered_df[filtered_df.apply(
            lambda x: isinstance(x.get('skills'), dict) and 
                     isinstance(x['skills'].get('experience'), dict) and 
                     x['skills']['experience'].get('role') in filters['roles'],
            axis=1
        )]
    
    return filtered_df

def show_overview_tab(filtered_df):
    """Display the Overview tab content."""
    st.markdown("### Survey Overview")
    
    # Key Metrics with Trends
    with st.container():
        st.subheader("Key Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            total_responses = len(filtered_df)
            st.metric(
                "Total Responses",
                total_responses,
                delta=None
            )
        
        with metrics_col2:
            try:
                years_list = []
                for _, row in filtered_df.iterrows():
                    years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                    if years > 0:
                        years_list.append(years)
                
                if years_list:
                    years_mean = round(sum(years_list) / len(years_list), 1)
                    st.metric("Average Years in Construction", years_mean)
                else:
                    st.metric("Average Years in Construction", "N/A")
            except Exception as e:
                st.metric("Average Years in Construction", "N/A")
        
        with metrics_col3:
            try:
                roles = []
                for _, row in filtered_df.iterrows():
                    role = row.get('skills', {}).get('experience', {}).get('role')
                    if role:
                        roles.append(role)
                
                if roles:
                    most_common = max(set(roles), key=roles.count)
                    st.metric("Most Common Role", most_common.replace('-', ' ').title())
                else:
                    st.metric("Most Common Role", "N/A")
            except Exception as e:
                st.metric("Most Common Role", "N/A")
        
        with metrics_col4:
            try:
                avg_skills = sum([len(row.get('skills', {}).get('technical', [])) 
                                for _, row in filtered_df.iterrows()]) / len(filtered_df)
                st.metric("Average Skills per Person", f"{avg_skills:.1f}")
            except Exception as e:
                st.metric("Average Skills per Person", "N/A")
    
    # Response Trends
    st.subheader("Response Trends")
    try:
        dates = pd.to_datetime(filtered_df['submittedAt'])
        daily_counts = dates.dt.date.value_counts().sort_index()
        
        fig_trends = px.line(
            x=daily_counts.index,
            y=daily_counts.values,
            title="Daily Response Trends",
            labels={'x': 'Date', 'y': 'Number of Responses'}
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying response trends: {str(e)}")
    
    # Demographics Overview
    st.subheader("Demographics Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Experience Distribution
            exp_ranges = []
            for _, row in filtered_df.iterrows():
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                if years < 2:
                    exp_ranges.append("0-2 years")
                elif years < 5:
                    exp_ranges.append("2-5 years")
                elif years < 10:
                    exp_ranges.append("5-10 years")
                else:
                    exp_ranges.append("10+ years")
            
            exp_df = pd.DataFrame(exp_ranges, columns=['Experience Range'])
            fig_exp = px.pie(
                exp_df,
                names='Experience Range',
                title='Experience Distribution'
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying experience distribution: {str(e)}")
    
    with col2:
        try:
            # Project Size Distribution
            sizes = [row.get('skills', {}).get('experience', {}).get('projectSize', 'Unknown')
                    for _, row in filtered_df.iterrows()]
            size_df = pd.DataFrame(sizes, columns=['Project Size'])
            fig_size = px.pie(
                size_df,
                names='Project Size',
                title='Project Size Distribution'
            )
            st.plotly_chart(fig_size, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying project size distribution: {str(e)}")
    
    # Career Goals and Preferences
    st.subheader("Career Development Overview")
    col3, col4 = st.columns(2)
    
    with col3:
        try:
            # Career Goals Distribution
            goals_list = []
            for _, row in filtered_df.iterrows():
                goals = row.get('goals', {}).get('careerGoals', [])
                if isinstance(goals, list):
                    goals_list.extend(goals)
            
            if goals_list:
                goals_df = pd.DataFrame(goals_list, columns=['Goal'])
                fig_goals = px.pie(
                    goals_df,
                    names='Goal',
                    title='Career Goals Distribution'
                )
                st.plotly_chart(fig_goals, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying career goals: {str(e)}")
    
    with col4:
        try:
            # Advancement Preferences
            prefs = []
            for _, row in filtered_df.iterrows():
                pref = row.get('goals', {}).get('advancementPreference')
                if pref:
                    prefs.append(pref)
            
            if prefs:
                prefs_df = pd.DataFrame(prefs, columns=['Preference'])
                fig_prefs = px.pie(
                    prefs_df,
                    names='Preference',
                    title='Advancement Preferences'
                )
                st.plotly_chart(fig_prefs, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying advancement preferences: {str(e)}")
    
    # Skills Overview
    st.subheader("Skills Overview")
    try:
        # Technical Skills Distribution
        skills_data = []
        for _, row in filtered_df.iterrows():
            skills = row.get('skills', {}).get('technical', [])
            if isinstance(skills, list):
                skills_data.extend(skills)
        
        if skills_data:
            skills_df = pd.DataFrame(pd.Series(skills_data).value_counts()).reset_index()
            skills_df.columns = ['Skill', 'Count']
            
            fig_skills = px.bar(
                skills_df,
                x='Skill',
                y='Count',
                title='Technical Skills Distribution'
            )
            fig_skills.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_skills, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying skills overview: {str(e)}")
    
    # Personality Overview
    st.subheader("Personality Type Overview")
    col5, col6 = st.columns(2)
    
    with col5:
        try:
            # MBTI Distribution
            mbti_data = []
            for _, row in filtered_df.iterrows():
                traits = row.get('personalityTraits', {}).get('myersBriggs', {})
                if isinstance(traits, dict):
                    mbti_type = ''
                    for trait in ['attention', 'information', 'decisions', 'lifestyle']:
                        if trait in traits and traits[trait]:
                            mbti_type += traits[trait][0]
                    if len(mbti_type) == 4:
                        mbti_data.append(mbti_type)
            
            if mbti_data:
                # Create DataFrame and get value counts
                mbti_counts = pd.DataFrame(pd.Series(mbti_data).value_counts()).reset_index()
                mbti_counts.columns = ['MBTI Type', 'Count']  # Rename columns
                
                fig_mbti = px.pie(
                    mbti_counts,
                    values='Count',
                    names='MBTI Type',  # Use the renamed column
                    title='MBTI Type Distribution'
                )
                st.plotly_chart(fig_mbti, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying MBTI distribution: {str(e)}")
    
    with col6:
        try:
            # Holland Code Distribution
            holland_data = []
            for _, row in filtered_df.iterrows():
                codes = row.get('personalityTraits', {}).get('hollandCode', [])
                if isinstance(codes, list):
                    holland_data.extend(codes)
            
            if holland_data:
                # Create DataFrame and get value counts
                holland_counts = pd.DataFrame(pd.Series(holland_data).value_counts()).reset_index()
                holland_counts.columns = ['Holland Code', 'Count']  # Rename columns
                
                fig_holland = px.pie(
                    holland_counts,
                    values='Count',
                    names='Holland Code',  # Use the renamed column
                    title='Holland Code Distribution'
                )
                st.plotly_chart(fig_holland, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying Holland code distribution: {str(e)}")

def show_analytics_tab(filtered_df):
    """Display the Analytics tab content."""
    st.markdown("### Advanced Analytics")
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Personality Clusters", "Career Progression", "Skills Analysis", "ML Insights"],
        horizontal=True
    )
    
    if analysis_type == "Personality Clusters":
        st.subheader("Personality Type Analysis")
        
        try:
            # MBTI Analysis
            st.write("### MBTI Analysis")
            mbti_data = []
            roles = []
            years_exp = []
            for _, row in filtered_df.iterrows():
                traits = row.get('personalityTraits', {}).get('myersBriggs', {})
                role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                
                if isinstance(traits, dict):
                    mbti_type = ''
                    individual_traits = []
                    for trait in ['attention', 'information', 'decisions', 'lifestyle']:
                        if trait in traits and traits[trait]:
                            value = traits[trait][0]
                            mbti_type += value
                            individual_traits.append(value)
                    if len(mbti_type) == 4:
                        mbti_data.append({
                            'MBTI': mbti_type,
                            'Attention': individual_traits[0],
                            'Information': individual_traits[1],
                            'Decisions': individual_traits[2],
                            'Lifestyle': individual_traits[3],
                            'Role': role,
                            'Years': years
                        })
            
            if mbti_data:
                mbti_df = pd.DataFrame(mbti_data)
                
                # MBTI Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Type Distribution Pie Chart
                    type_counts = mbti_df['MBTI'].value_counts()
                    fig1 = px.pie(values=type_counts.values,
                                names=type_counts.index,
                                title='MBTI Type Distribution')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Individual Trait Distribution
                    trait_data = []
                    for trait in ['Attention', 'Information', 'Decisions', 'Lifestyle']:
                        counts = mbti_df[trait].value_counts()
                        for val, count in counts.items():
                            trait_data.append({
                                'Trait': trait,
                                'Value': val,
                                'Count': count
                            })
                    
                    trait_df = pd.DataFrame(trait_data)
                    fig2 = px.bar(trait_df,
                                x='Trait',
                                y='Count',
                                color='Value',
                                title='MBTI Trait Distribution',
                                barmode='group')
                    st.plotly_chart(fig2, use_container_width=True)
                
                # MBTI by Experience Level
                st.write("### MBTI Types by Experience")
                
                # Calculate average years for each MBTI type
                avg_years = mbti_df.groupby('MBTI')['Years'].mean().sort_values(ascending=False)
                fig3 = px.bar(avg_years,
                             title='Average Years of Experience by MBTI Type')
                st.plotly_chart(fig3, use_container_width=True)
                
                # MBTI-Role Analysis
                st.write("### MBTI Types by Role")
                
                # Create heatmap of MBTI types vs roles
                role_mbti = pd.crosstab(mbti_df['MBTI'], mbti_df['Role'])
                fig4 = px.imshow(role_mbti,
                               title='MBTI Type Distribution Across Roles',
                               aspect='auto')
                st.plotly_chart(fig4, use_container_width=True)
            
            # Holland Code Analysis
            st.write("### Holland Code Analysis")
            holland_data = []
            for _, row in filtered_df.iterrows():
                codes = row.get('personalityTraits', {}).get('hollandCode', [])
                role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                
                if isinstance(codes, list) and codes:
                    for code in codes:
                        holland_data.append({
                            'Code': code,
                            'Role': role,
                            'Years': years
                        })
            
            if holland_data:
                holland_df = pd.DataFrame(holland_data)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Holland Code Distribution
                    code_counts = holland_df['Code'].value_counts()
                    fig5 = px.pie(values=code_counts.values,
                                names=code_counts.index,
                                title='Holland Code Distribution')
                    st.plotly_chart(fig5, use_container_width=True)
                
                with col4:
                    # Average Experience by Holland Code
                    avg_years = holland_df.groupby('Code')['Years'].mean().sort_values(ascending=False)
                    fig6 = px.bar(avg_years,
                                title='Average Years of Experience by Holland Code')
                    st.plotly_chart(fig6, use_container_width=True)
                
                # Holland Code-Role Analysis
                role_holland = pd.crosstab(holland_df['Code'], holland_df['Role'])
                fig7 = px.imshow(role_holland,
                               title='Holland Code Distribution Across Roles',
                               aspect='auto')
                st.plotly_chart(fig7, use_container_width=True)
                
                # Summary Statistics
                st.write("### Personality Summary")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    most_common_mbti = mbti_df['MBTI'].mode()[0]
                    st.metric("Most Common MBTI Type", most_common_mbti)
                
                with col6:
                    most_common_holland = holland_df['Code'].mode()[0]
                    st.metric("Most Common Holland Code", most_common_holland)
                
                with col7:
                    avg_exp = mbti_df['Years'].mean()
                    st.metric("Average Years of Experience", f"{avg_exp:.1f}")
        
        except Exception as e:
            st.error(f"Error in personality analysis: {str(e)}")
    
    elif analysis_type == "Career Progression":
        st.subheader("Career Progression Analysis")
        
        try:
            # Experience vs Role Box Plot
            exp_data = []
            for _, row in filtered_df.iterrows():
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                salary = row.get('goals', {}).get('targetSalary', 'entry')
                project_size = row.get('skills', {}).get('experience', {}).get('projectSize', 'Unknown')
                
                if role != 'Unknown':
                    exp_data.append({
                        'Role': role.replace('-', ' ').title(),
                        'Years': years,
                        'Salary Level': salary.title(),
                        'Project Size': project_size.title()
                    })
            
            if exp_data:
                exp_df = pd.DataFrame(exp_data)
                
                # Experience Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box Plot
                    fig1 = px.box(exp_df, x='Role', y='Years', 
                                title='Years of Experience by Role')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Violin Plot
                    fig2 = px.violin(exp_df, x='Role', y='Years',
                                   title='Experience Distribution by Role',
                                   box=True)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Project Size Analysis
                st.subheader("Project Size Analysis")
                col3, col4 = st.columns(2)
                
                with col3:
                    # Project Size Distribution
                    size_counts = exp_df['Project Size'].value_counts()
                    fig3 = px.pie(values=size_counts.values, 
                                names=size_counts.index,
                                title='Project Size Distribution')
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col4:
                    # Average Experience by Project Size
                    avg_exp = exp_df.groupby('Project Size')['Years'].mean().reset_index()
                    fig4 = px.bar(avg_exp, x='Project Size', y='Years',
                                title='Average Experience by Project Size')
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Salary Level Analysis
                st.subheader("Salary Level Analysis")
                col5, col6 = st.columns(2)
                
                with col5:
                    # Experience vs Salary Level
                    fig5 = px.box(exp_df, x='Salary Level', y='Years',
                                title='Experience Distribution by Salary Level')
                    st.plotly_chart(fig5, use_container_width=True)
                
                with col6:
                    # Role Distribution by Salary Level
                    role_salary = pd.crosstab(exp_df['Role'], exp_df['Salary Level'])
                    fig6 = px.imshow(role_salary,
                                   title='Role Distribution by Salary Level',
                                   aspect='auto')
                    st.plotly_chart(fig6, use_container_width=True)
                
                # Career Path Sankey (existing code)
                st.subheader("Career Path Flow")
                path_data = []
                for _, row in filtered_df.iterrows():
                    current = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                    target = row.get('workPreferences', {}).get('roles', ['Unknown'])[0]
                    salary = row.get('goals', {}).get('targetSalary', 'entry')
                    
                    if current != 'Unknown' and target != 'Unknown':
                        path_data.append({
                            'Current': current.replace('-', ' ').title(),
                            'Target': target.replace('-', ' ').title(),
                            'Salary': salary.title()
                        })
                
                if path_data:
                    path_df = pd.DataFrame(path_data)
                    fig7 = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=list(set(path_df['Current'].unique()).union(
                                   path_df['Target'].unique()).union(
                                   path_df['Salary'].unique())),
                            color="lightblue"
                        ),
                        link=dict(
                            source=[path_df['Current'].unique().tolist().index(x) for x in path_df['Current']] +
                                   [len(path_df['Current'].unique()) + path_df['Target'].unique().tolist().index(x) 
                                    for x in path_df['Target']],
                            target=[len(path_df['Current'].unique()) + path_df['Target'].unique().tolist().index(x) 
                                   for x in path_df['Target']] +
                                   [len(path_df['Current'].unique()) + len(path_df['Target'].unique()) + 
                                    path_df['Salary'].unique().tolist().index(x) for x in path_df['Salary']],
                            value=[1] * (len(path_df) * 2)
                        )
                    )])
                    fig7.update_layout(title_text="Career Path Flow")
                    st.plotly_chart(fig7, use_container_width=True)
                
                # Summary Statistics
                st.subheader("Career Progression Summary")
                col7, col8, col9 = st.columns(3)
                
                with col7:
                    avg_exp = exp_df['Years'].mean()
                    st.metric("Average Years of Experience", f"{avg_exp:.1f}")
                
                with col8:
                    most_common_role = exp_df['Role'].mode()[0]
                    st.metric("Most Common Role", most_common_role)
                
                with col9:
                    most_common_salary = exp_df['Salary Level'].mode()[0]
                    st.metric("Most Common Salary Level", most_common_salary)
        
        except Exception as e:
            st.error(f"Error in career progression analysis: {str(e)}")
    
    elif analysis_type == "Skills Analysis":
        st.subheader("Skills Analysis")
        try:
            # Extract skills data
            skills_data = []
            certifications_data = []
            for _, row in filtered_df.iterrows():
                skills = row.get('skills', {}).get('technical', [])
                certs = row.get('skills', {}).get('certifications', [])
                if isinstance(skills, list):
                    skills_data.extend(skills)
                if isinstance(certs, list):
                    certifications_data.extend(certs)
            
            if skills_data:
                # Technical Skills Analysis
                st.write("### Technical Skills Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    skills_counts = pd.DataFrame(pd.Series(skills_data).value_counts()).reset_index()
                    skills_counts.columns = ['Skill', 'Count']
                    
                    fig1 = px.bar(skills_counts, 
                                x='Skill', 
                                y='Count',
                                title='Technical Skills Distribution')
                    fig1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig2 = px.pie(skills_counts,
                                values='Count',
                                names='Skill',
                                title='Skills Proportion')
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Skills by Experience Level
                st.write("### Skills by Experience Level")
                skills_by_exp = []
                for _, row in filtered_df.iterrows():
                    years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                    skills = row.get('skills', {}).get('technical', [])
                    if isinstance(skills, list):
                        exp_level = "Junior (0-2 years)" if years < 2 else \
                                  "Mid-level (2-5 years)" if years < 5 else \
                                  "Senior (5-10 years)" if years < 10 else \
                                  "Expert (10+ years)"
                        for skill in skills:
                            skills_by_exp.append({
                                'Skill': skill,
                                'Experience Level': exp_level
                            })
                
                if skills_by_exp:
                    exp_df = pd.DataFrame(skills_by_exp)
                    fig3 = px.histogram(exp_df,
                                      x='Experience Level',
                                      color='Skill',
                                      title='Skills Distribution by Experience Level',
                                      barmode='group')
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Certifications Analysis
                if certifications_data:
                    st.write("### Certifications Analysis")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Certifications distribution
                        cert_counts = pd.DataFrame(pd.Series(certifications_data).value_counts()).reset_index()
                        cert_counts.columns = ['Certification', 'Count']
                        
                        fig4 = px.bar(cert_counts,
                                    x='Certification',
                                    y='Count',
                                    title='Certifications Distribution')
                        fig4.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig4, use_container_width=True)
                    
                    with col4:
                        # Average number of certifications by experience
                        cert_by_exp = []
                        for _, row in filtered_df.iterrows():
                            years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                            certs = row.get('skills', {}).get('certifications', [])
                            if isinstance(certs, list):
                                exp_level = "Junior (0-2 years)" if years < 2 else \
                                          "Mid-level (2-5 years)" if years < 5 else \
                                          "Senior (5-10 years)" if years < 10 else \
                                          "Expert (10+ years)"
                                cert_by_exp.append({
                                    'Experience Level': exp_level,
                                    'Number of Certifications': len(certs)
                                })
                        
                        cert_exp_df = pd.DataFrame(cert_by_exp)
                        avg_certs = cert_exp_df.groupby('Experience Level')['Number of Certifications'].mean().reset_index()
                        
                        fig5 = px.bar(avg_certs,
                                    x='Experience Level',
                                    y='Number of Certifications',
                                    title='Average Certifications by Experience Level')
                        st.plotly_chart(fig5, use_container_width=True)
                
                # Skills Co-occurrence Analysis
                st.write("### Skills Relationships")
                # Create co-occurrence matrix
                all_skills = list(set(skills_data))
                matrix = np.zeros((len(all_skills), len(all_skills)))
                
                for _, row in filtered_df.iterrows():
                    skills = row.get('skills', {}).get('technical', [])
                    if isinstance(skills, list):
                        for i, skill1 in enumerate(all_skills):
                            for j, skill2 in enumerate(all_skills):
                                if skill1 in skills and skill2 in skills:
                                    matrix[i, j] += 1
                
                fig6 = px.imshow(matrix,
                               x=all_skills,
                               y=all_skills,
                               title="Skills Co-occurrence Matrix",
                               color_continuous_scale="Viridis")
                st.plotly_chart(fig6, use_container_width=True)
                
                # Summary Statistics
                st.write("### Skills Summary")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    avg_skills = len(skills_data) / len(filtered_df)
                    st.metric("Average Skills per Person", f"{avg_skills:.1f}")
                
                with col6:
                    most_common_skill = skills_counts['Skill'].iloc[0]
                    st.metric("Most Common Skill", most_common_skill)
                
                with col7:
                    if certifications_data:
                        avg_certs = len(certifications_data) / len(filtered_df)
                        st.metric("Average Certifications", f"{avg_certs:.1f}")
                    else:
                        st.metric("Average Certifications", "N/A")
            
            else:
                st.info("No skills data available")
            
        except Exception as e:
            st.error(f"Error in skills analysis: {str(e)}")
    
    elif analysis_type == "ML Insights":
        st.subheader("Machine Learning Insights")
        
        try:
            # Experience Prediction Model
            st.write("### Experience Level Prediction")
            
            # Prepare data
            exp_data = []
            for _, row in filtered_df.iterrows():
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                skills = len(row.get('skills', {}).get('technical', []))
                salary = row.get('goals', {}).get('targetSalary', 'entry')
                
                exp_data.append({
                    'Years': years,
                    'Role': role,
                    'Skills': skills,
                    'Salary': salary
                })
            
            if exp_data:
                exp_df = pd.DataFrame(exp_data)
                
                # Create correlation heatmap
                corr_matrix = exp_df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix,
                              title="Feature Correlation Heatmap",
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
                
                # Clustering Analysis
                if len(exp_df) > 5:  # Need enough data for clustering
                    # Prepare features for clustering
                    features = exp_df[['Years', 'Skills']].values
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    # Add clusters to dataframe
                    exp_df['Cluster'] = clusters
                    
                    # Create scatter plot
                    fig = px.scatter(exp_df,
                                   x='Years',
                                   y='Skills',
                                   color='Cluster',
                                   title='Experience Clusters',
                                   labels={'Years': 'Years in Construction',
                                          'Skills': 'Number of Skills'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Personality Analysis with ML
            st.write("### Personality Type Analysis")
            
            # Extract MBTI data
            mbti_data = []
            roles = []
            for _, row in filtered_df.iterrows():
                traits = row.get('personalityTraits', {}).get('myersBriggs', {})
                role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                
                if isinstance(traits, dict) and role != 'Unknown':
                    mbti_type = ''
                    for trait in ['attention', 'information', 'decisions', 'lifestyle']:
                        if trait in traits and traits[trait]:
                            mbti_type += traits[trait][0]
                    if len(mbti_type) == 4:
                        mbti_data.append(mbti_type)
                        roles.append(role)
            
            if mbti_data and roles:
                # Create personality-role correlation
                personality_df = pd.DataFrame({
                    'MBTI': mbti_data,
                    'Role': roles
                })
                
                # Create cross-tabulation
                cross_tab = pd.crosstab(personality_df['MBTI'], personality_df['Role'])
                
                # Create heatmap
                fig = px.imshow(cross_tab,
                              title="Personality-Role Distribution",
                              color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            # Add PCA Analysis for Personality Types
            st.write("### Personality Type PCA Analysis")
            try:
                # Prepare personality data
                personality_data = []
                for _, row in filtered_df.iterrows():
                    traits = row.get('personalityTraits', {})
                    if isinstance(traits, dict):
                        # MBTI encoding
                        mbti = traits.get('myersBriggs', {})
                        if isinstance(mbti, dict):
                            mbti_features = []
                            for trait in ['attention', 'information', 'decisions', 'lifestyle']:
                                value = mbti.get(trait, [''])[0]
                                mbti_features.append(1 if value in ['E', 'S', 'T', 'J'] else 0)
                            
                            # Holland code encoding
                            holland = traits.get('hollandCode', [])
                            holland_features = [1 if code in holland else 0 
                                             for code in HOLLAND_CODES]
                            
                            personality_data.append(mbti_features + holland_features)
            
                if personality_data:
                    # Perform PCA
                    pca = PCA(n_components=2)
                    personality_transformed = pca.fit_transform(personality_data)
                    
                    # Create PCA plot
                    pca_df = pd.DataFrame(
                        personality_transformed,
                        columns=['PC1', 'PC2']
                    )
                    
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        title='Personality Type PCA',
                        labels={'PC1': 'First Principal Component',
                               'PC2': 'Second Principal Component'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show explained variance
                    explained_var = pca.explained_variance_ratio_
                    st.write(f"Explained variance ratio: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
            
            except Exception as e:
                st.error(f"Error in PCA analysis: {str(e)}")
            
            # Experience Prediction with Regression
            st.write("### Experience Level Regression")
            try:
                if len(exp_df) > 10:  # Need enough data for regression
                    # Prepare features
                    X = exp_df[['Skills']].values
                    y = exp_df['Years'].values
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Fit regression
                    reg = LinearRegression()
                    reg.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = reg.predict(X_test)
                    
                    # Create regression plot
                    fig = go.Figure()
                    
                    # Add actual data points
                    fig.add_trace(go.Scatter(
                        x=X_test.flatten(),
                        y=y_test,
                        mode='markers',
                        name='Actual',
                        marker=dict(color='blue')
                    ))
                    
                    # Add regression line
                    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_line = reg.predict(X_line)
                    
                    fig.add_trace(go.Scatter(
                        x=X_line.flatten(),
                        y=y_line,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title='Skills vs Years Experience Regression',
                        xaxis_title='Number of Skills',
                        yaxis_title='Years of Experience'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show regression metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    st.write(f"R² Score: {r2:.2f}")
                    st.write(f"Mean Squared Error: {mse:.2f}")
            
            except Exception as e:
                st.error(f"Error in regression analysis: {str(e)}")
            
            # Skills Clustering Dendrogram
            st.write("### Skills Clustering Dendrogram")
            try:
                # Extract skills data
                skills_data = []
                for _, row in filtered_df.iterrows():
                    skills = row.get('skills', {}).get('technical', [])
                    if isinstance(skills, list):
                        skills_data.extend(skills)
                
                if skills_data:
                    # Create skills co-occurrence matrix
                    skills_set = list(set(skills_data))
                    n_skills = len(skills_set)
                    skills_matrix = np.zeros((n_skills, n_skills))
                    
                    for _, row in filtered_df.iterrows():
                        skills = row.get('skills', {}).get('technical', [])
                        if isinstance(skills, list):
                            for i, skill1 in enumerate(skills_set):
                                for j, skill2 in enumerate(skills_set):
                                    if skill1 in skills and skill2 in skills:
                                        skills_matrix[i, j] += 1
                    
                    # Create linkage matrix
                    linkage_matrix = linkage(skills_matrix, method='ward')
                    
                    # Create dendrogram
                    fig = ff.create_dendrogram(
                        skills_matrix,
                        labels=skills_set,
                        orientation='left',
                        linkagefun=lambda x: linkage_matrix
                    )
                    
                    fig.update_layout(
                        title='Skills Clustering Dendrogram',
                        width=800,
                        height=500 + len(skills_set) * 20
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in dendrogram analysis: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in ML analysis: {str(e)}")

def show_data_tab(filtered_df):
    """Display the Data tab content."""
    st.markdown("### Raw Data Explorer")
    
    # Add data download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="survey_data.csv",
        mime="text/csv",
    )
    
    # Add search/filter options
    search_term = st.text_input("Search in data")
    if search_term:
        filtered_df = filtered_df[filtered_df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False).any(), axis=1
        )]
    
    st.dataframe(filtered_df)

# Then: Main code
with st.spinner("Loading survey data..."):
    df = get_survey_data()

# Create and apply filters
filters = create_sidebar_filters(df)
filtered_df = df.copy()

try:
    filtered_df = apply_filters(df, filters)
except Exception as e:
    st.error(f"Error applying filters: {str(e)}")

# Create tabs
tabs = st.tabs(["Overview", "Analytics", "Data"])

# Tab content
with tabs[0]:
    try:
        show_overview_tab(filtered_df)
    except Exception as e:
        st.error(f"Error in Overview tab: {str(e)}")

with tabs[1]:
    try:
        show_analytics_tab(filtered_df)
    except Exception as e:
        st.error(f"Error in Analytics tab: {str(e)}")

with tabs[2]:
    try:
        show_data_tab(filtered_df)
    except Exception as e:
        st.error(f"Error in Data tab: {str(e)}")

# Add footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #6c757d;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #dee2e6;
}
.footer a {
    color: #1E67C7;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    <p>© Embers Staffing 2025 | Built by <a href="https://bigfootcrane.com/" target="_blank">Bigfoot Crane</a></p>
</div>
""", unsafe_allow_html=True) 