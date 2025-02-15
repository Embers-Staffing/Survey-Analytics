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

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

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
    # Debug info
    st.write("Data shape:", filtered_df.shape)
    st.write("Columns:", filtered_df.columns.tolist())
    
    # Sample data structure
    if not filtered_df.empty:
        st.write("Sample data structure:")
        sample_row = filtered_df.iloc[0].to_dict()
        st.json(sample_row)
    
    # Check nested structures
    st.write("Data validation:")
    data_checks = {
        "Skills data": sum([1 for row in filtered_df.itertuples() if isinstance(getattr(row, 'skills', None), dict)]),
        "Goals data": sum([1 for row in filtered_df.itertuples() if isinstance(getattr(row, 'goals', None), dict)]),
        "Personality data": sum([1 for row in filtered_df.itertuples() if isinstance(getattr(row, 'personalityTraits', None), dict)]),
        "Work preferences": sum([1 for row in filtered_df.itertuples() if isinstance(getattr(row, 'workPreferences', None), dict)])
    }
    st.write(data_checks)

    # Key Metrics
    with st.container():
        st.subheader("Key Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(
                "Total Responses",
                len(filtered_df),
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
                st.error(f"Years calculation error: {str(e)}")
        
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
                st.error(f"Role calculation error: {str(e)}")

    # Career Development
    with st.expander("Career Development Analysis", expanded=True):
        st.subheader("Career Goals and Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                goals_list = []
                for _, row in filtered_df.iterrows():
                    goals = row.get('goals', {}).get('careerGoals', [])
                    if isinstance(goals, list):
                        goals_list.extend(goals)
                
                if goals_list:
                    career_goals_df = pd.DataFrame(goals_list, columns=['goal'])
                    fig = px.pie(career_goals_df, names='goal', 
                               title='Career Goals Distribution',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No career goals data available")
            except Exception as e:
                st.write("Error displaying career goals chart")
        
        with col2:
            try:
                prefs = []
                for _, row in filtered_df.iterrows():
                    pref = row.get('goals', {}).get('advancementPreference')
                    if pref:
                        prefs.append(pref)
                
                if prefs:
                    pref_df = pd.DataFrame(prefs, columns=['preference'])
                    fig = px.pie(pref_df, names='preference',
                               title='Advancement Preferences',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No advancement preference data available")
            except Exception as e:
                st.write("Error displaying preferences chart")

    # Career Path Analysis
    st.subheader("Career Path Analysis")
    try:
        # Create role progression data
        role_progression = []
        for _, row in filtered_df.iterrows():
            try:
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                current_role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                target_roles = row.get('workPreferences', {}).get('roles', ['Unknown'])
                target_role = target_roles[0] if target_roles else 'Unknown'
                salary_level = row.get('goals', {}).get('targetSalary', 'entry')
                
                # Clean up role names
                current_role = current_role.replace('-', ' ').title()
                target_role = target_role.replace('-', ' ').title()
                
                role_progression.append({
                    'Current Role': current_role,
                    'Target Role': target_role,
                    'Salary Level': salary_level.title()
                })
            except:
                continue
        
        if role_progression:
            # Create Sankey diagram
            prog_df = pd.DataFrame(role_progression)
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = list(set(prog_df['Current Role'].unique()).union(
                               prog_df['Target Role'].unique()).union(
                               prog_df['Salary Level'].unique())),
                    color = "lightblue"
                ),
                link = dict(
                    source = [prog_df['Current Role'].unique().tolist().index(x) for x in prog_df['Current Role']] +
                            [len(prog_df['Current Role'].unique()) + prog_df['Target Role'].unique().tolist().index(x) 
                             for x in prog_df['Target Role']],
                    target = [len(prog_df['Current Role'].unique()) + prog_df['Target Role'].unique().tolist().index(x) 
                             for x in prog_df['Target Role']] +
                            [len(prog_df['Current Role'].unique()) + len(prog_df['Target Role'].unique()) + 
                             prog_df['Salary Level'].unique().tolist().index(x) for x in prog_df['Salary Level']],
                    value = [1] * (len(prog_df) * 2)
                )
            )])
            
            fig.update_layout(title_text="Career Progression Flow")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No career progression data available")
            
    except Exception as e:
        st.error(f"Error displaying career progression: {str(e)}")

    # Skills Analysis
    st.subheader("Skills Distribution")
    try:
        # Extract technical skills
        all_skills = []
        for _, row in filtered_df.iterrows():
            skills = row.get('skills', {}).get('technical', [])
            if isinstance(skills, list):
                all_skills.extend(skills)
        
        if all_skills:
            # Create DataFrame and get value counts
            skills_counts = pd.DataFrame(
                pd.Series(all_skills).value_counts()
            ).reset_index()
            skills_counts.columns = ['Skill', 'Count']  # Rename columns
            
            # Create bar chart
            fig = px.bar(
                skills_counts,
                x='Skill',
                y='Count',
                title='Technical Skills Distribution'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Skill",
                yaxis_title="Number of Responses",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skills data available")
            
    except Exception as e:
        st.error(f"Error displaying skills distribution: {str(e)}")

def show_analytics_tab(filtered_df):
    """Display the Analytics tab content."""
    st.markdown("### Advanced Analytics")
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Personality Clusters", "Career Progression", "Skills Analysis"],
        horizontal=True
    )
    
    if analysis_type == "Personality Clusters":
        st.subheader("Personality Type Analysis")
        try:
            # Extract MBTI data
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
                # Create DataFrame with value counts
                mbti_counts = pd.DataFrame(
                    pd.Series(mbti_data).value_counts()
                ).reset_index()
                mbti_counts.columns = ['MBTI Type', 'Count']  # Rename columns
                
                # Create pie chart
                fig = px.pie(
                    mbti_counts,
                    values='Count',
                    names='MBTI Type',
                    title='MBTI Type Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show type breakdown
                st.write("### MBTI Type Breakdown")
                st.dataframe(mbti_counts)
            else:
                st.info("No personality data available")
                
        except Exception as e:
            st.error(f"Error analyzing personality data: {str(e)}")
    
    elif analysis_type == "Career Progression":
        st.subheader("Career Progression Analysis")
        try:
            # Create experience vs role chart
            exp_data = []
            for _, row in filtered_df.iterrows():
                years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                if role != 'Unknown':
                    exp_data.append({
                        'Role': role.replace('-', ' ').title(),
                        'Years': years
                    })
            
            if exp_data:
                exp_df = pd.DataFrame(exp_data)
                fig = px.box(exp_df,
                           x='Role',
                           y='Years',
                           title='Years of Experience by Role')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No career progression data available")
                
        except Exception as e:
            st.error(f"Error analyzing career progression: {str(e)}")
    
    else:  # Skills Analysis
        st.subheader("Skills Analysis")
        try:
            # Create skills correlation matrix
            skills_data = []
            for _, row in filtered_df.iterrows():
                skills = row.get('skills', {}).get('technical', [])
                if isinstance(skills, list):
                    skills_data.append(skills)
            
            if skills_data:
                # Create co-occurrence matrix
                all_skills = list(set([skill for skills in skills_data for skill in skills]))
                matrix = np.zeros((len(all_skills), len(all_skills)))
                
                for skills in skills_data:
                    for i, skill1 in enumerate(all_skills):
                        for j, skill2 in enumerate(all_skills):
                            if skill1 in skills and skill2 in skills:
                                matrix[i, j] += 1
                
                fig = px.imshow(matrix,
                              x=all_skills,
                              y=all_skills,
                              title="Skills Co-occurrence Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No skills data available")
                
        except Exception as e:
            st.error(f"Error analyzing skills: {str(e)}")

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
</style>
<div class="footer">
    <p>© Embers Staffing 2025 | Built by <a href="https://github.com/ArsCodeAmatoria" target="_blank">ArsCodeAmatoria</a></p>
</div>
""", unsafe_allow_html=True) 