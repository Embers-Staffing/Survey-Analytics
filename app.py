import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

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
    responses_ref = db.collection('responses')
    docs = responses_ref.stream()
    
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    
    return pd.DataFrame(data)

def create_sidebar_filters(df: pd.DataFrame) -> dict:
    """Create sidebar filters for the dashboard."""
    st.sidebar.header("Filters")
    
    # Date Range Filter with better error handling
    if 'submittedAt' in df.columns:
        try:
            # Convert to datetime with error handling
            dates = pd.to_datetime(df['submittedAt'], errors='coerce', utc=True)
            valid_dates = dates.dropna()
            
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
                st.sidebar.warning("No valid dates found in the data")
        except Exception as e:
            st.sidebar.error(f"Error processing dates: {str(e)}")
            date_range = None
    else:
        date_range = None
    
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
    
    return {
        'date_range': date_range,
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

# Main dashboard code
if check_password():
    # Load data
    with st.spinner("Loading survey data..."):
        df = get_survey_data()
    
    # Create filters
    filters = create_sidebar_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Dashboard Tabs with better styling and organization
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>""", unsafe_allow_html=True)

    tabs = st.tabs([
        "Overview",
        "Analytics",
        "Data"
    ])

    with tabs[0]:  # Overview Tab
        st.markdown("### Survey Overview")
        st.markdown("""
        <style>
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Key Metrics in a nice box
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            color: rgb(30, 103, 119);
            overflow-wrap: break-word;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        with st.container():
            st.subheader("Key Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric(
                    "Total Responses",
                    len(filtered_df),
                    delta=f"{len(filtered_df) - len(df)} filtered"
                )
            
            with metrics_col2:
                try:
                    years_list = []
                    for _, row in filtered_df.iterrows():
                        personal_info = row['personalInfo']
                        if isinstance(personal_info, dict):
                            year_str = str(personal_info.get('yearsInConstruction', '0'))
                            # Remove any non-numeric characters
                            year_str = ''.join(c for c in year_str if c.isdigit())
                            if year_str:
                                years_list.append(float(year_str))
                    
                    if years_list:
                        years_mean = round(sum(years_list) / len(years_list), 1)
                        st.metric("Average Years in Construction", years_mean)
                    else:
                        st.metric("Average Years in Construction", "N/A")
                except Exception as e:
                    st.metric("Average Years in Construction", "N/A")
                    st.write(f"Error calculating years: {str(e)}")
            
            with metrics_col3:
                try:
                    roles = []
                    for _, row in filtered_df.iterrows():
                        skills = row['skills']
                        if isinstance(skills, dict) and 'experience' in skills:
                            experience = skills['experience']
                            if isinstance(experience, dict) and 'role' in experience:
                                roles.append(experience['role'])
                    
                    if roles:
                        role_counts = {}
                        for role in roles:
                            role_counts[role] = role_counts.get(role, 0) + 1
                        
                        # Get most common role
                        most_common = max(role_counts.items(), key=lambda x: x[1])[0]
                        st.metric("Most Common Role", most_common.replace('-', ' ').title())
                    else:
                        st.metric("Most Common Role", "N/A")
                except Exception as e:
                    st.metric("Most Common Role", "N/A")
                    st.write(f"Error calculating roles: {str(e)}")

        # Career Development
        with st.expander("Career Development Analysis", expanded=True):
            st.subheader("Career Goals and Preferences")
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    goals_list = []
                    for _, row in filtered_df.iterrows():
                        goals = row['goals']
                        if isinstance(goals, dict) and 'careerGoals' in goals:
                            career_goals = goals['careerGoals']
                            if isinstance(career_goals, list):
                                goals_list.extend(career_goals)
                    
                    if goals_list:
                        career_goals_df = pd.DataFrame(goals_list, columns=['goal'])
                        fig = px.pie(career_goals_df, names='goal', 
                                   title='Career Goals Distribution',
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("No career goals data available")
                except Exception as e:
                    st.write("No career goals data available")
                    st.write(f"Error: {str(e)}")
            
            with col2:
                try:
                    prefs = []
                    for _, row in filtered_df.iterrows():
                        goals = row['goals']
                        if isinstance(goals, dict) and 'advancementPreference' in goals:
                            prefs.append(goals['advancementPreference'])
                    
                    if prefs:
                        pref_df = pd.DataFrame(prefs, columns=['preference'])
                        fig = px.pie(pref_df, names='preference',
                                   title='Advancement Preferences',
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("No advancement preference data available")
                except Exception as e:
                    st.write("No advancement preference data available")
                    st.write(f"Error: {str(e)}")

            # Career Path Visualization
            st.subheader("Career Path Analysis")
            
            # Create Sankey diagram of role progression
            role_progression = []
            for _, row in filtered_df.iterrows():
                try:
                    years = float(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                    current_role = row.get('skills', {}).get('experience', {}).get('role', 'Unknown')
                    target_roles = row.get('workPreferences', {}).get('roles', ['Unknown'])
                    target_role = target_roles[0] if target_roles else 'Unknown'
                    
                    # Clean up role names
                    current_role = current_role.replace('-', ' ').title()
                    target_role = target_role.replace('-', ' ').title()
                    
                    role_progression.append({
                        'Years': years,
                        'Current Role': current_role,
                        'Target Role': target_role
                    })
                except:
                    continue
            
            if role_progression:
                prog_df = pd.DataFrame(role_progression)
                
                # Create Sankey diagram with improved styling
                source = []
                target = []
                value = []
                
                # Get unique roles and create node labels
                all_roles = sorted(set(prog_df['Current Role'].unique()) | set(prog_df['Target Role'].unique()))
                
                # Create links
                for current in prog_df['Current Role'].unique():
                    for target_role in prog_df['Target Role'].unique():
                        count = len(prog_df[
                            (prog_df['Current Role'] == current) & 
                            (prog_df['Target Role'] == target_role)
                        ])
                        if count > 0:
                            source.append(all_roles.index(current))
                            target.append(all_roles.index(target_role))
                            value.append(count)
                
                # Create Sankey diagram with improved styling
                fig = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 40,  # Increased padding
                        thickness = 25,  # Adjusted thickness
                        line = dict(color = "black", width = 0.5),
                        label = all_roles,
                        color = px.colors.qualitative.Set3,  # Better color scheme
                        font = dict(size = 14, color = "black")  # Larger, clearer font
                    ),
                    link = dict(
                        source = source,
                        target = target,
                        value = value,
                        color = [f'rgba(44, 160, 44, {v/max(value):.2f})' for v in value]  # Improved transparency
                    )
                )])
                
                # Update layout for better visibility
                fig.update_layout(
                    title = dict(
                        text = "Career Progression Paths",
                        font = dict(size=24),
                        y = 0.95
                    ),
                    font_size = 14,
                    height = 800,  # Increased height
                    margin = dict(t=60, l=20, r=20, b=20),
                    paper_bgcolor = 'white',
                    plot_bgcolor = 'white'
                )
                
                # Show diagram
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                st.subheader("Career Transition Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Most Common Current Roles:")
                    current_role_counts = prog_df['Current Role'].value_counts().head(3)
                    for role, count in current_role_counts.items():
                        st.write(f"- {role}: {count} respondents")
                
                with col2:
                    st.write("Most Desired Target Roles:")
                    target_role_counts = prog_df['Target Role'].value_counts().head(3)
                    for role, count in target_role_counts.items():
                        st.write(f"- {role}: {count} respondents")
                
                # Add explanation
                st.info("""
                This Sankey diagram shows career progression paths from current roles (left) 
                to target roles (right). The width of each flow indicates the number of 
                respondents following that path. Hover over the diagram for detailed information.
                """)

        # Skills Analysis
        with st.expander("Skills Analysis", expanded=True):
            st.subheader("Technical Skills and Experience")
            try:
                # Extract technical skills
                all_skills = [
                    skill 
                    for _, row in filtered_df.iterrows()
                    for skill in row.get('skills', {}).get('technical', [])
                    if isinstance(row.get('skills', {}), dict)
                ]
                
                if all_skills:
                    skills_df = pd.DataFrame(all_skills, columns=['Skill'])
                    
                    # Add visualization selector
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        ["Distribution", "Heatmap", "Time Trends", "Network"],
                        key="skills_viz_selector_overview"
                    )
                    
                    # Add interactive filters at the top
                    st.subheader("Filter Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_count = st.slider(
                            "Minimum Skill Count",
                            min_value=1,
                            max_value=int(skills_df['Skill'].value_counts().max()),
                            value=1,
                            key="skill_count_filter"
                        )
                    
                    with col2:
                        selected_skills = st.multiselect(
                            "Select Specific Skills",
                            options=sorted(skills_df['Skill'].unique()),
                            default=[],
                            key="skill_selector"
                        )
                    
                    # Filter data based on selections
                    skill_counts = skills_df['Skill'].value_counts()
                    filtered_skills = skill_counts[skill_counts >= min_count]
                    
                    if selected_skills:
                        filtered_skills = filtered_skills[filtered_skills.index.isin(selected_skills)]
                    
                    if viz_type == "Distribution":
                        # Enhanced Distribution charts
                        st.subheader("Skills Distribution")
                        
                        # Interactive Histogram
                        fig1 = px.histogram(
                            skills_df[skills_df['Skill'].isin(filtered_skills.index)],
                            x='Skill',
                            title='Technical Skills Distribution',
                            color_discrete_sequence=['#2ecc71'],
                            hover_data=['Skill'],
                            labels={'count': 'Number of Responses', 'Skill': 'Technical Skill'}
                        )
                        fig1.update_layout(
                            showlegend=True,
                            hovermode='x unified',
                            hoverlabel=dict(bgcolor="white"),
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig1, use_container_width=True, key="skills_hist_overview")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Enhanced Pie Chart
                            fig2 = px.pie(
                                values=filtered_skills.values,
                                names=filtered_skills.index,
                                title='Skills Distribution (Pie Chart)',
                                hover_data=[filtered_skills.values],
                                custom_data=[filtered_skills.values/len(skills_df)*100]
                            )
                            fig2.update_traces(
                                textposition='inside',
                                hovertemplate="<b>%{label}</b><br>" +
                                            "Count: %{value}<br>" +
                                            "Percentage: %{customdata[0]:.1f}%"
                            )
                            st.plotly_chart(fig2, use_container_width=True, key="skills_pie_overview")
                        
                        with col2:
                            # Enhanced Bar Chart
                            fig3 = px.bar(
                                x=filtered_skills.index,
                                y=filtered_skills.values,
                                title='Skills Distribution (Bar Chart)',
                                labels={'x': 'Skill', 'y': 'Count'},
                                color=filtered_skills.values,
                                color_continuous_scale='Viridis'
                            )
                            fig3.update_layout(
                                xaxis_tickangle=-45,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig3, use_container_width=True, key="skills_bar_overview")
                        
                        # Interactive Summary Table
                        st.subheader("Skills Summary")
                        summary_df = pd.DataFrame({
                            'Skill': filtered_skills.index,
                            'Count': filtered_skills.values,
                            'Percentage': (filtered_skills.values / len(skills_df) * 100).round(1)
                        }).sort_values('Count', ascending=False)
                        
                        # Add sorting options
                        sort_by = st.selectbox(
                            "Sort by",
                            options=['Count', 'Skill', 'Percentage'],
                            key="summary_sort"
                        )
                        ascending = st.checkbox("Ascending order", key="summary_order")
                        
                        summary_df = summary_df.sort_values(sort_by, ascending=ascending)
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            column_config={
                                "Percentage": st.column_config.NumberColumn(
                                    "Percentage",
                                    help="Percentage of total responses",
                                    format="%.1f%%"
                                )
                            }
                        )
                    elif viz_type == "Heatmap":
                        st.subheader("Skills Co-occurrence Heatmap")
                        
                        # Group skills by response
                        skill_groups = []
                        for _, row in filtered_df.iterrows():
                            skills = row.get('skills', {}).get('technical', [])
                            if isinstance(skills, list):
                                skill_groups.append(skills)
                        
                        # Create co-occurrence matrix
                        all_unique_skills = list(set([s for group in skill_groups for s in group]))
                        co_occurrence = np.zeros((len(all_unique_skills), len(all_unique_skills)))
                        
                        for group in skill_groups:
                            for i, skill1 in enumerate(all_unique_skills):
                                for j, skill2 in enumerate(all_unique_skills):
                                    if skill1 in group and skill2 in group:
                                        co_occurrence[i, j] += 1
                        
                        # Create heatmap
                        fig = px.imshow(
                            co_occurrence,
                            x=all_unique_skills,
                            y=all_unique_skills,
                            title="Skill Co-occurrence Matrix",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="skills_heatmap_overview")
                        
                        # Add explanation
                        st.info("This heatmap shows how often different skills appear together in responses. " +
                               "Darker colors indicate skills that are more frequently found together.")
                    elif viz_type == "Time Trends":
                        st.subheader("Skills Adoption Over Time")
                        
                        # Create time series data
                        time_data = []
                        for _, row in filtered_df.iterrows():
                            date = pd.to_datetime(row.get('submittedAt'))
                            skills = row.get('skills', {}).get('technical', [])
                            if isinstance(skills, list):
                                for skill in skills:
                                    time_data.append({
                                        'Date': date,
                                        'Skill': skill
                                    })
                        
                        if time_data:
                            time_df = pd.DataFrame(time_data)
                            time_df = time_df.sort_values('Date')
                            
                            # Cumulative adoption chart
                            fig = px.line(
                                time_df.groupby(['Date', 'Skill']).size().unstack().cumsum(),
                                title="Skill Adoption Trends",
                                labels={'value': 'Number of Users', 'variable': 'Skill'}
                            )
                            st.plotly_chart(fig, use_container_width=True, key="skills_trends_overview")
                            
                            # Monthly adoption rate
                            monthly_adoption = time_df.groupby([pd.Grouper(key='Date', freq='M'), 'Skill']).size().unstack()
                            fig2 = px.bar(
                                monthly_adoption,
                                title="Monthly Skill Adoption Rate",
                                labels={'value': 'New Users', 'variable': 'Skill'}
                            )
                            st.plotly_chart(fig2, use_container_width=True, key="skills_monthly_overview")
                            
                            st.info("These charts show how skill adoption has grown over time and the monthly adoption rate.")
                    elif viz_type == "Network":
                        st.subheader("Skills Relationship Network")
                        
                        # Create skill pairs for network
                        skill_pairs = []
                        pair_weights = {}
                        for _, row in filtered_df.iterrows():
                            skills = row.get('skills', {}).get('technical', [])
                            if isinstance(skills, list) and len(skills) > 1:
                                for i in range(len(skills)):
                                    for j in range(i+1, len(skills)):
                                        pair = tuple(sorted([skills[i], skills[j]]))
                                        skill_pairs.append(pair)
                                        pair_weights[pair] = pair_weights.get(pair, 0) + 1
                        
                        if skill_pairs:
                            # Create network data
                            nodes = list(set([skill for pair in skill_pairs for skill in pair]))
                            edges = list(pair_weights.items())
                            
                            # Create network visualization
                            edge_x = []
                            edge_y = []
                            for edge, weight in edges:
                                x0, y0 = nodes.index(edge[0]) * 2, nodes.index(edge[0]) * 2
                                x1, y1 = nodes.index(edge[1]) * 2, nodes.index(edge[1]) * 2
                                edge_x.extend([x0, x1, None])
                                edge_y.extend([y0, y1, None])
                            
                            node_x = [i * 2 for i in range(len(nodes))]
                            node_y = [i * 2 for i in range(len(nodes))]
                            
                            # Create the network plot
                            fig = go.Figure()
                            
                            # Add edges
                            fig.add_trace(go.Scatter(
                                x=edge_x, y=edge_y,
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none',
                                mode='lines'
                            ))
                            
                            # Add nodes
                            fig.add_trace(go.Scatter(
                                x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                text=nodes,
                                textposition="top center",
                                marker=dict(
                                    size=20,
                                    color='#2ecc71',
                                    line_width=2
                                )
                            ))
                            
                            fig.update_layout(
                                title="Skills Relationship Network",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="skills_network_overview")
                            
                            # Show top relationships
                            st.subheader("Top Skill Relationships")
                            top_pairs = sorted(pair_weights.items(), key=lambda x: x[1], reverse=True)[:5]
                            for pair, weight in top_pairs:
                                st.write(f"- {pair[0]} + {pair[1]}: {weight} occurrences")
                            
                            st.info("The network graph shows how different skills are related. " +
                                   "Connected skills frequently appear together in responses.")
                    else:
                        st.info("No technical skills found in the data")
            except Exception as e:
                st.error(f"Skills analysis error: {str(e)}")
                st.write("No technical skills data available")

    with tabs[1]:  # Analytics Tab
        st.markdown("### Advanced Analytics")
        
        # Analytics sections in expandable containers
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Personality Clusters", "Career Progression", "Skills Analysis"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if analysis_type == "Personality Clusters":
            with st.container():
                st.subheader("Personality Clustering")
                try:
                    # Create features for clustering
                    features_data = []
                    for _, row in filtered_df.iterrows():
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
                    
                    # K-Means Clustering Analysis
                    st.subheader("Personality Cluster Analysis")
                    
                    # Prepare data for clustering
                    cluster_data = []
                    for _, row in filtered_df.iterrows():
                        try:
                            # Get numeric features with strict validation
                            years_str = str(row.get('personalInfo', {}).get('yearsInConstruction', '0'))
                            years = float(''.join(c for c in years_str if c.isdigit()) or '0')
                            
                            project_size_map = {
                                'small': 1,
                                'medium': 2,
                                'large': 3
                            }
                            project_size = project_size_map.get(
                                row.get('skills', {}).get('experience', {}).get('projectSize', 'small'),
                                1  # Default to small if missing
                            )
                            
                            # Get personality type with validation
                            mbti = row.get('personalityTraits', {}).get('myersBriggs', {})
                            if not isinstance(mbti, dict):
                                mbti = {}
                            
                            # Calculate personality score with defaults
                            personality_score = 0
                            for trait_type in ['attention', 'information', 'decisions', 'lifestyle']:
                                trait_list = mbti.get(trait_type, [''])
                                if trait_list and isinstance(trait_list, list):
                                    trait = trait_list[0]
                                    personality_score += 1 if trait in ['E', 'S', 'T', 'J'] else 0
                            
                            # Only add if we have valid numeric values
                            if years >= 0 and project_size in [1, 2, 3]:
                                cluster_data.append({
                                    'Years': years,
                                    'Project Size': project_size,
                                    'Personality Score': personality_score
                                })
                        except Exception as e:
                            continue  # Skip invalid entries
                    
                    if len(cluster_data) >= 3:  # Only proceed if we have enough valid data
                        cluster_df = pd.DataFrame(cluster_data)
                        
                        # Standardize features
                        X_cluster = StandardScaler().fit_transform(cluster_df)
                        
                        # Perform K-means clustering
                        from sklearn.cluster import KMeans
                        n_clusters = 3
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_cluster)
                        
                        # Add clusters to dataframe
                        cluster_df['Cluster'] = clusters
                        
                        # Create 3D scatter plot
                        fig = px.scatter_3d(
                            cluster_df,
                            x='Years',
                            y='Project Size',
                            z='Personality Score',
                            color='Cluster',
                            title='Personality Clusters',
                            labels={
                                'Years': 'Years of Experience',
                                'Project Size': 'Project Size (1=Small, 3=Large)',
                                'Personality Score': 'Personality Type Score'
                            },
                            color_continuous_scale='Viridis'
                        )
                        
                        # Update layout
                        fig.update_layout(
                            scene = dict(
                                xaxis_title='Years of Experience',
                                yaxis_title='Project Size',
                                zaxis_title='Personality Score'
                            ),
                            height=700
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add cluster analysis
                        st.subheader("Cluster Analysis")
                        
                        # Calculate cluster statistics
                        cluster_stats = cluster_df.groupby('Cluster').agg({
                            'Years': ['mean', 'count'],
                            'Project Size': 'mean',
                            'Personality Score': 'mean'
                        }).round(2)
                        
                        # Format cluster statistics
                        cluster_stats.columns = [
                            'Avg Years', 'Count', 'Avg Project Size', 'Avg Personality Score'
                        ]
                        st.dataframe(cluster_stats)
                        
                        # Add interpretation
                        st.info("""
                        Cluster Analysis Interpretation:
                        - Each point represents a survey respondent
                        - Colors indicate different personality clusters
                        - Clusters are based on years of experience, project size, and personality type
                        - Similar profiles are grouped together
                        - Hover over points to see detailed information
                        """)
                        
                        # Add cluster descriptions
                        st.subheader("Cluster Profiles")
                        for i in range(n_clusters):
                            stats = cluster_stats.iloc[i]
                            st.write(f"**Cluster {i}** ({stats['Count']} members):")
                            st.write(f"- Average Experience: {stats['Avg Years']:.1f} years")
                            st.write(f"- Typical Project Size: {stats['Avg Project Size']:.1f}")
                            st.write(f"- Personality Score: {stats['Avg Personality Score']:.1f}")
                            st.write("")
                    
                except Exception as e:
                    st.error(f"Clustering error: {str(e)}")
                    st.write("Unable to generate personality cluster visualization")
            
            # Personality Insights
            st.subheader("Personality Type Distribution")
            try:
                # Extract MBTI data
                mbti_types = []
                for _, row in filtered_df.iterrows():
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

        elif analysis_type == "Career Progression":
            with st.container():
                st.subheader("Career Progression Analysis")
                try:
                    # Prepare data for regression
                    regression_data = []
                    for _, row in filtered_df.iterrows():
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
                    
                    # Create heatmap using px.imshow
                    fig = px.imshow(
                        corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        color_continuous_scale='RdBu_r',
                        title='Correlation Matrix'
                    )
                    
                    # Update layout for better readability
                    fig.update_traces(text=corr.values.round(2), texttemplate='%{text}')
                    fig.update_layout(
                        xaxis_title='',
                        yaxis_title='',
                        xaxis={'side': 'bottom'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # PCA Analysis
                    st.subheader("PCA Analysis")
                    
                    # Prepare data for PCA
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    
                    # Define features for PCA
                    X = regression_df[['Years', 'Project Size']].values  # Define X here
                    y = regression_df['Target Salary Level'].values
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform PCA
                    pca = PCA()
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Calculate explained variance ratio
                    explained_variance = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    # Create subplot figure
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(
                            'Explained Variance Ratio',
                            'Cumulative Explained Variance'
                        )
                    )
                    
                    # Add bar plot for explained variance
                    fig.add_trace(
                        go.Bar(
                            x=[f'PC{i+1}' for i in range(len(explained_variance))],
                            y=explained_variance,
                            name='Explained Variance',
                            marker_color='rgb(55, 83, 109)'
                        ),
                        row=1, col=1
                    )
                    
                    # Add line plot for cumulative variance
                    fig.add_trace(
                        go.Scatter(
                            x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                            y=cumulative_variance,
                            name='Cumulative Variance',
                            mode='lines+markers',
                            marker=dict(color='rgb(26, 118, 255)')
                        ),
                        row=1, col=2
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=400,
                        showlegend=True,
                        title_text="PCA Explained Variance Analysis"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add feature importance based on PCA
                    st.subheader("Feature Importance from PCA")
                    
                    # Calculate feature importance
                    feature_importance = np.abs(pca.components_[0])
                    feature_names = ['Years', 'Project Size']
                    
                    # Create feature importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=True)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance from First Principal Component',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Absolute Coefficient Value',
                        yaxis_title='Feature',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.info("""
                    PCA Analysis Interpretation:
                    - Bars show how much variance each principal component explains
                    - Line shows cumulative explained variance
                    - Higher importance values indicate more influential features
                    - First principal component shows the most important patterns in the data
                    """)
                    
                    # Decision Boundary Plot
                    st.subheader("Decision Boundary Analysis")
                    
                    # Prepare data for decision boundary plot
                    X = regression_df[['Years', 'Project Size']].values
                    y = regression_df['Target Salary Level'].values
                    
                    # Dynamically set n_neighbors based on dataset size
                    n_neighbors = min(5, len(X) - 1)  # Ensure n_neighbors is less than n_samples
                    
                    if len(X) > n_neighbors:  # Only proceed if we have enough samples
                        # Create mesh grid
                        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                        xx, yy = np.meshgrid(
                            np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1)
                        )
                        
                        # Fit a simple classifier with dynamic n_neighbors
                        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
                        clf.fit(X, y)
                        
                        # Make predictions on mesh grid
                        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        
                        # Create contour plot
                        fig = go.Figure()
                        
                        # Add contour
                        fig.add_trace(
                            go.Contour(
                                x=np.arange(x_min, x_max, 0.1),
                                y=np.arange(y_min, y_max, 0.1),
                                z=Z,
                                colorscale='Viridis',
                                showscale=True,
                                name='Decision Boundary'
                            )
                        )
                        
                        # Add scatter points
                        for salary_level in np.unique(y):
                            mask = y == salary_level
                            fig.add_trace(
                                go.Scatter(
                                    x=X[mask, 0],
                                    y=X[mask, 1],
                                    mode='markers',
                                    name=f'Salary Level {salary_level}',
                                    marker=dict(size=8)
                                )
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title='Decision Boundaries between Salary Levels',
                            xaxis_title='Years of Experience',
                            yaxis_title='Project Size',
                            height=600,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add interpretation
                        st.info(f"""
                        This analysis uses {n_neighbors} nearest neighbors to classify salary levels based on:
                        - Years of Experience
                        - Project Size
                        
                        The colored regions show predicted salary levels for different combinations 
                        of experience and project size.
                        """)
                    else:
                        st.warning("Not enough data points for decision boundary analysis. Need at least 2 samples.")
                    
                    # Residual Plot Analysis
                    st.subheader("Regression Residual Analysis")
                    
                    # Fit a simple linear regression for salary prediction
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Make predictions
                    y_pred = model.predict(X)
                    
                    # Calculate residuals
                    residuals = y - y_pred
                    
                    # Create residual plots
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            'Residuals vs Predicted',
                            'Residuals vs Years',
                            'Residuals Distribution',
                            'Residuals vs Project Size'
                        )
                    )
                    
                    # Residuals vs Predicted
                    fig.add_trace(
                        go.Scatter(
                            x=y_pred,
                            y=residuals,
                            mode='markers',
                            name='Residuals',
                            marker=dict(color='blue', size=8)
                        ),
                        row=1, col=1
                    )
                    
                    # Residuals vs Years
                    fig.add_trace(
                        go.Scatter(
                            x=X[:, 0],
                            y=residuals,
                            mode='markers',
                            name='Residuals',
                            marker=dict(color='green', size=8)
                        ),
                        row=1, col=2
                    )
                    
                    # Residuals Distribution
                    fig.add_trace(
                        go.Histogram(
                            x=residuals,
                            name='Residuals',
                            nbinsx=20,
                            marker_color='red'
                        ),
                        row=2, col=1
                    )
                    
                    # Residuals vs Project Size
                    fig.add_trace(
                        go.Scatter(
                            x=X[:, 1],
                            y=residuals,
                            mode='markers',
                            name='Residuals',
                            marker=dict(color='purple', size=8)
                        ),
                        row=2, col=2
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=800,
                        title_text="Regression Residual Analysis",
                        showlegend=False
                    )
                    
                    # Update axes labels
                    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
                    fig.update_xaxes(title_text="Years", row=1, col=2)
                    fig.update_xaxes(title_text="Residuals", row=2, col=1)
                    fig.update_xaxes(title_text="Project Size", row=2, col=2)
                    
                    fig.update_yaxes(title_text="Residuals", row=1, col=1)
                    fig.update_yaxes(title_text="Residuals", row=1, col=2)
                    fig.update_yaxes(title_text="Count", row=2, col=1)
                    fig.update_yaxes(title_text="Residuals", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add model metrics
                    st.subheader("Regression Model Metrics")
                    from sklearn.metrics import r2_score, mean_squared_error
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "R² Score",
                            f"{r2_score(y, y_pred):.3f}"
                        )
                        st.metric(
                            "RMSE",
                            f"{np.sqrt(mean_squared_error(y, y_pred)):.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "R² Score (Training)",
                            f"{r2_score(y, y_pred):.3f}"
                        )
                        st.metric(
                            "RMSE (Training)",
                            f"{np.sqrt(mean_squared_error(y, y_pred)):.3f}"
                        )
                    
                    # Add interpretation
                    st.info("""
                    Residual Plot Interpretation:
                    - Random scatter around zero indicates a good fit
                    - Patterns suggest model assumptions may be violated
                    - Normal distribution of residuals is ideal
                    - R² closer to 1 indicates better model fit
                    - Lower RMSE indicates better prediction accuracy
                    """)
                    
                except Exception as e:
                    st.error(f"Regression analysis error: {str(e)}")
                    st.write("Unable to perform regression analysis")

        else:  # Skills Analysis
            with st.container():
                st.subheader("Skills Distribution")
                try:
                    # Extract technical skills
                    all_skills = [
                        skill 
                        for _, row in filtered_df.iterrows()
                        for skill in row.get('skills', {}).get('technical', [])
                        if isinstance(row.get('skills', {}), dict)
                    ]
                    
                    if all_skills:
                        skills_df = pd.DataFrame(all_skills, columns=['Skill'])
                        
                        # Skills Distribution Chart
                        fig1 = px.histogram(
                            skills_df,
                            x='Skill',
                            title='Technical Skills Distribution',
                            color_discrete_sequence=['#2ecc71']
                        )
                        st.plotly_chart(fig1, use_container_width=True, key="skills_hist")
                        
                        # Add Skills Summary
                        st.subheader("Skills Breakdown")
                        skill_counts = skills_df['Skill'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Pie Chart
                            fig2 = px.pie(
                                values=skill_counts.values,
                                names=skill_counts.index,
                                title='Skills Distribution (Pie Chart)'
                            )
                            st.plotly_chart(fig2, use_container_width=True, key="skills_pie")
                        
                        with col2:
                            # Bar Chart
                            fig3 = px.bar(
                                x=skill_counts.index,
                                y=skill_counts.values,
                                title='Skills Distribution (Bar Chart)',
                                labels={'x': 'Skill', 'y': 'Count'}
                            )
                            st.plotly_chart(fig3, use_container_width=True, key="skills_bar")
                        
                        # Skills Summary Table
                        st.subheader("Skills Summary")
                        summary_df = pd.DataFrame({
                            'Skill': skill_counts.index,
                            'Count': skill_counts.values,
                            'Percentage': (skill_counts.values / len(skills_df) * 100).round(1)
                        })
                        st.dataframe(summary_df, use_container_width=True)
                        
                    else:
                        st.info("No technical skills found in the data")
                except Exception as e:
                    st.error(f"Skills analysis error: {str(e)}")
                    st.write("No technical skills data available")

    with tabs[2]:  # Data Tab
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