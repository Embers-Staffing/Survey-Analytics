# Construction Career Survey Analytics Dashboard

A data analytics platform for visualizing and analyzing construction industry career survey responses.

## Overview

This dashboard provides insights into construction industry career paths, skills, and trends using data collected from industry professionals. Built with Streamlit and Firebase, it offers interactive visualizations and advanced analytics.

## Features

- Real-time data visualization
- Career progression analysis
- Skills distribution tracking
- Personality type clustering
- Interactive filtering and sorting
- Advanced analytics and trend analysis

## Technology Stack

- Python 3.x
- Streamlit
- Firebase/Firestore
- Plotly
- Pandas
- Scikit-learn

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Embers-Staffing/Survey-Analytics.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Firebase:
   - Create a Firebase project
   - Set up Firestore database
   - Create `.streamlit/secrets.toml` with the following structure:
     ```toml
     password = "your_dashboard_password"
     
     [firebase]
     type = "service_account"
     project_id = "your_project_id"
     private_key_id = "your_private_key_id"
     private_key = "your_private_key"
     client_email = "your_client_email"
     client_id = "your_client_id"
     auth_uri = "https://accounts.google.com/o/oauth2/auth"
     token_uri = "https://oauth2.googleapis.com/token"
     auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
     client_x509_cert_url = "your_client_cert_url"
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application and dashboard logic
- `auth.py`: Authentication handling
- `generate_dummy_data.py`: Test data generation
- `.streamlit/`: Configuration files
- `requirements.txt`: Project dependencies

## Security

- Password-protected dashboard access
- Secure credential management
- Data encryption in transit
- Regular security updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a pull request

## License

Copyright © 2025 Embers Staffing Solutions. All rights reserved.

## Deployment

1. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
2. Add your credentials to `.streamlit/secrets.toml`
3. For Streamlit Cloud deployment, add these secrets in your app settings:
   ```toml
   password = "9aba7500"
   
   [firebase]
   # ... your firebase credentials
   ``` 