import firebase_admin
from firebase_admin import credentials, firestore
import random
from datetime import datetime, timedelta
import streamlit as st
import json

# Get credentials from Streamlit secrets and save to file
creds = {
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

# Save credentials to file
with open('firebase-credentials.json', 'w') as f:
    json.dump(creds, f, indent=4)

# Initialize Firebase
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase-credentials.json')
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    print("Successfully connected to Firebase!")

except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    exit(1)

# Sample data for generating realistic responses
ROLES = ['project-manager', 'site-supervisor', 'engineer', 'architect', 'estimator', 'safety-manager']
PROJECT_SIZES = ['small', 'medium', 'large']
TECHNICAL_SKILLS = [
    'AutoCAD', 'Revit', 'SketchUp', 'MS Project', 'Primavera P6', 
    'BIM', 'Blueprint Reading', 'Cost Estimation', 'Scheduling',
    'OSHA Safety', 'Quality Control', 'Contract Management'
]
MBTI_TYPES = {
    'attention': ['E', 'I'],
    'information': ['S', 'N'],
    'decisions': ['T', 'F'],
    'lifestyle': ['J', 'P']
}
HOLLAND_CODES = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
ENVIRONMENTS = ['office', 'field', 'hybrid']
CAREER_GOALS = ['leadership', 'technical-expertise', 'project-management', 'business-development']
SALARY_LEVELS = ['entry', 'mid', 'senior', 'executive']

def generate_dummy_response():
    """Generate a single dummy response."""
    # Generate a random datetime within the last year
    now = datetime.now()
    random_days = random.randint(0, 365)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    submission_date = (now - timedelta(days=random_days, hours=random_hours, minutes=random_minutes))
    
    return {
        'submittedAt': submission_date.isoformat(),  # Use ISO format for consistent datetime
        'personalInfo': {
            'yearsInConstruction': str(random.randint(0, 30)),
            'location': random.choice(['BC', 'AB', 'ON', 'QC'])
        },
        'skills': {
            'technical': random.sample(TECHNICAL_SKILLS, random.randint(3, 8)),
            'experience': {
                'role': random.choice(ROLES),
                'projectSize': random.choice(PROJECT_SIZES)
            }
        },
        'personalityTraits': {
            'myersBriggs': {
                'attention': [random.choice(MBTI_TYPES['attention'])],
                'information': [random.choice(MBTI_TYPES['information'])],
                'decisions': [random.choice(MBTI_TYPES['decisions'])],
                'lifestyle': [random.choice(MBTI_TYPES['lifestyle'])]
            },
            'hollandCode': random.sample(HOLLAND_CODES, 3)
        },
        'workPreferences': {
            'environment': random.choice(ENVIRONMENTS),
            'roles': random.sample(ROLES, random.randint(1, 3)),
            'travelWillingness': random.choice(['low', 'medium', 'high'])
        },
        'goals': {
            'careerGoals': random.sample(CAREER_GOALS, random.randint(1, 3)),
            'targetSalary': random.choice(SALARY_LEVELS),
            'advancementPreference': random.choice(['technical', 'managerial'])
        }
    }

def upload_dummy_data(num_responses=100):
    """Generate and upload dummy responses to Firebase."""
    responses_ref = db.collection('responses')
    
    for _ in range(num_responses):
        dummy_response = generate_dummy_response()
        responses_ref.add(dummy_response)
        
    print(f"Successfully uploaded {num_responses} dummy responses to Firebase")

if __name__ == "__main__":
    # Generate and upload 100 dummy responses
    upload_dummy_data(100) 