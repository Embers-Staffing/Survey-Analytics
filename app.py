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
import plotly.graph_objects as go
from datetime import datetime

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

def create_sidebar_filters(df):
    """Create sidebar filters for the dashboard."""
    st.sidebar.header("Filters")
    
    # Date Range Filter
    if 'submittedAt' in df.columns:
        dates = pd.to_datetime(df['submittedAt'])
        min_date = dates.min()
        max_date = dates.max()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
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
        mask = (pd.to_datetime(filtered_df['submittedAt']).dt.date >= filters['date_range'][0]) & \
               (pd.to_datetime(filtered_df['submittedAt']).dt.date <= filters['date_range'][1])
        filtered_df = filtered_df[mask]
    
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
                        most_common = max(role_counts.items(), key=lambda x: x[1])[0]
                        st.metric("Most Common Role", most_common)
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
                node_colors = px.colors.qualitative.Set3[:len(all_roles)]
                
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
                        pad = 20,
                        thickness = 30,
                        line = dict(color = "black", width = 0.5),
                        label = all_roles,
                        color = node_colors
                    ),
                    link = dict(
                        source = source,
                        target = target,
                        value = value,
                        color = [f'rgba(44, 160, 44, {v/max(value)})' for v in value]
                    )
                )])
                
                # Update layout
                fig.update_layout(
                    title_text="Career Progression Paths",
                    font_size=12,
                    height=600,
                    margin=dict(t=40, l=0, r=0, b=0)
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
                all_skills = []
                for _, row in filtered_df.iterrows():
                    skills = row.get('skills', {})
                    if isinstance(skills, dict):
                        tech = skills.get('technical', [])
                        if isinstance(tech, list):
                            all_skills.extend(tech)
                
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

        else:  # Skills Analysis
            with st.container():
                st.subheader("Skills Distribution")
                try:
                    # Extract technical skills
                    all_skills = []
                    for _, row in filtered_df.iterrows():
                        skills = row.get('skills', {})
                        if isinstance(skills, dict):
                            tech = skills.get('technical', [])
                            if isinstance(tech, list):
                                all_skills.extend(tech)
                    
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