import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import google.generativeai as genai
import os
from dotenv import load_dotenv


# Configure Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.write("API Key Loaded:", os.getenv("GOOGLE_API_KEY") is not None)

# Function to classify risk level
def classify_risk(score):
    if score <= 0.2:
        return "🟢 LOW", "Minimal service stress - operations running smoothly"
    elif score <= 0.4:
        return "🟡 MODERATE", "Moderate service stress - monitor closely"
    elif score <= 0.7:
        return "🟠 HIGH", "High service stress - intervention recommended"
    else:
        return "🔴 CRITICAL", "Critical service stress - immediate action required"

# Function to generate why explanation
def generate_why_explanation(row, df):
    bio = row['biometric_to_enrolment_ratio']
    child = row['child_update_pressure'] if pd.notna(row['child_update_pressure']) else 0
    elderly = row['elderly_update_pressure'] if pd.notna(row['elderly_update_pressure']) else 0
    
    reasons = []
    if bio > df['biometric_to_enrolment_ratio'].quantile(0.75):
        reasons.append("high demand for biometric updates compared to new enrolments")
    if child > df['child_update_pressure'].quantile(0.75):
        reasons.append("significant pressure from child population updates")
    if elderly > df['elderly_update_pressure'].quantile(0.75):
        reasons.append("challenges in updating elderly citizens' biometric data")
    
    if not reasons:
        return "This district shows relatively low stress indicators across all measured areas."
    
    return f"This district is risky because of {', '.join(reasons)}. These factors contribute to service overload and potential delays in Aadhaar operations."

# Function to generate AI policy recommendation
@st.cache_data
def generate_ai_recommendation(state, district, risk_score, category, bio_ratio, child_pressure, elderly_pressure):
    prompt = f"""
You are a public policy advisor for the Government of India, specializing in Aadhaar ecosystem management.

District Context:
- State: {state}
- District: {district}
- Service Stress Risk Score: {risk_score:.4f}
- Risk Category: {category}
- Biometric-to-Enrolment Ratio: {bio_ratio:.2f}
- Child Update Pressure: {child_pressure:.4f}
- Elderly Update Pressure: {elderly_pressure:.4f}

Please provide a concise policy recommendation (150-200 words) that:
1. Explains WHY this district is experiencing service stress
2. Recommends 2-3 concrete administrative actions
3. Uses professional, government-friendly language
4. Avoids technical jargon

Structure your response with:
- Brief explanation of stress factors
- Specific recommendations
- Expected outcomes
"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate recommendation: {str(e)}"

# Function to compute model reliability metrics
@st.cache_data
def compute_model_metrics(df, _model):
    # Expected feature order from model training
    expected_features = ['age_0_5', 'age_5_17', 'age_18_greater', 'demo_age_5_17', 'demo_age_17_', 'bio_age_5_17', 'bio_age_17_', 'total_enrolment', 'total_biometric_updates', 'biometric_to_enrolment_ratio', 'elderly_update_pressure', 'child_update_pressure']
    
    # Select and reorder features to match training order
    features = df[expected_features]
    y_true = df['service_stress_risk']
    
    # Drop rows with NaN in features or target
    valid_mask = features.notna().all(axis=1) & y_true.notna()
    features = features[valid_mask]
    y_true = y_true[valid_mask]
    
    # Get predictions
    y_pred = _model.predict(features)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    spearman_corr, _ = scipy.stats.spearmanr(y_true, y_pred)
    
    # Top-K Risk Stability (overlap in top 20)
    top_k = 20
    true_top_indices = y_true.nlargest(top_k).index
    pred_top_indices = pd.Series(y_pred, index=y_true.index).nlargest(top_k).index
    overlap = len(set(true_top_indices) & set(pred_top_indices))
    stability = (overlap / top_k) * 100
    
    return mae, rmse, spearman_corr, stability

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('aadhaar_merged_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    # Handle missing values by dropping rows with NaN in key columns
    df = df.dropna(subset=['service_stress_risk', 'biometric_to_enrolment_ratio'])
    return df

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('aadhaar_service_stress_model.pkl')
    return model

df = load_data()
model = load_model()
metrics = compute_model_metrics(df, model)

# Sidebar Filters
st.sidebar.header('Dashboard Filters')

# State selection
states = sorted(df['state'].unique())
selected_state = st.sidebar.selectbox('Select State', states)

# District selection filtered by state
districts = sorted(df[df['state'] == selected_state]['district'].unique())
selected_district = st.sidebar.selectbox('Select District', districts)

# Date selection
dates = sorted(df['date'].dt.date.unique())
selected_date = st.sidebar.selectbox('Select Date', dates)

# Filter data for selected district and date
filtered_df = df[(df['state'] == selected_state) & 
                 (df['district'] == selected_district) & 
                 (df['date'].dt.date == selected_date)]

# Main Dashboard
st.title('Aadhaar Service Stress Risk Dashboard')
st.markdown('Interactive dashboard for judges to explore Aadhaar service stress risks across districts.')

if not filtered_df.empty:
    row = filtered_df.iloc[0]
    
    # District Risk Verdict
    st.header('District Risk Verdict')
    verdict, description = classify_risk(row['service_stress_risk'])
    
    # Determine colors based on risk level
    if 'LOW' in verdict:
        bg_color = '#d4edda'
        border_color = '#28a745'
        text_color = '#155724'
    elif 'MODERATE' in verdict:
        bg_color = '#fff3cd'
        border_color = '#ffc107'
        text_color = '#856404'
    elif 'HIGH' in verdict:
        bg_color = '#f8d7da'
        border_color = '#fd7e14'
        text_color = '#721c24'
    else:  # CRITICAL
        bg_color = '#f5c6cb'
        border_color = '#dc3545'
        text_color = '#721c24'
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color}; margin: 10px 0;">
    <h2 style="color: {text_color}; margin: 0; font-size: 24px;">{verdict}</h2>
    <p style="font-size: 18px; margin: 10px 0 0 0; color: {text_color};">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # District Risk Comparison
    st.header('District Risk Comparison')
    district_means = df.groupby('district')['service_stress_risk'].mean()
    selected_mean = district_means[selected_district]
    percentile = (district_means < selected_mean).mean() * 100
    st.metric('Risk Comparison', f"Riskier than {percentile:.1f}% of districts")
    
    st.divider()
    
    # Key Metrics
    st.header('Key Metrics for Selected District and Date')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Service Stress Risk Score', f"{row['service_stress_risk']:.4f}")
    
    with col2:
        st.metric('Biometric-to-Enrolment Ratio', f"{row['biometric_to_enrolment_ratio']:.2f}")
    
    with col3:
        child_pressure = row['child_update_pressure'] if pd.notna(row['child_update_pressure']) else 0
        st.metric('Child Update Pressure', f"{child_pressure:.4f}")
    
    with col4:
        elderly_pressure = row['elderly_update_pressure'] if pd.notna(row['elderly_update_pressure']) else 0
        st.metric('Elderly Update Pressure', f"{elderly_pressure:.4f}")
    
    st.divider()
    
    # Visualizations
    st.header('Service Stress Risk Trend Over Time')
    district_df = df[df['district'] == selected_district].sort_values('date')
    if not district_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(district_df['date'], district_df['service_stress_risk'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Service Stress Risk Score')
        ax.set_title(f'Service Stress Risk Trend for {selected_district}')
        st.pyplot(fig)
    else:
        st.write('No trend data available for this district.')
    
    st.divider()
    
    # Top 10 High-Risk Districts
    st.header('Top 10 High-Risk Districts (Overall Average Risk)')
    district_risk = df.groupby('district')['service_stress_risk'].mean().nlargest(10)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    district_risk.plot(kind='bar', ax=ax2, color='red')
    ax2.set_xlabel('District')
    ax2.set_ylabel('Average Service Stress Risk Score')
    ax2.set_title('Top 10 High-Risk Districts')
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.divider()
    
    # High-Risk Hotspots in Selected State
    st.header('High-Risk Hotspots in Selected State')
    state_risks = df[df['state'] == selected_state].groupby('district')['service_stress_risk'].mean().nlargest(5).reset_index()
    st.table(state_risks.rename(columns={'district': 'District', 'service_stress_risk': 'Average Risk Score'}))
    
    st.divider()
    
    # Explainability Section
    st.header('Risk Explanation')
    bio_ratio = row['biometric_to_enrolment_ratio']
    risk_score = row['service_stress_risk']
    
    if risk_score > df['service_stress_risk'].quantile(0.75):
        risk_level = 'high'
    elif risk_score > df['service_stress_risk'].quantile(0.5):
        risk_level = 'moderate'
    else:
        risk_level = 'low'
    
    if bio_ratio > df['biometric_to_enrolment_ratio'].quantile(0.75):
        bio_level = 'high'
    else:
        bio_level = 'manageable'
    
    explanation = f"This district exhibits {risk_level} service stress risk, primarily driven by a {bio_level} biometric-to-enrolment ratio. This suggests {'significant infrastructure and operational pressure' if bio_level == 'high' else 'relatively stable service demands'}."
    st.write(explanation)
    
    st.divider()
    
    # AI Policy Recommendation
    st.header('AI Policy Recommendation')
    
    if st.button('Generate AI Recommendation'):
        with st.spinner('Generating recommendation...'):
            ai_rec = generate_ai_recommendation(
                selected_state, 
                selected_district, 
                row['service_stress_risk'], 
                verdict.split()[0], 
                row['biometric_to_enrolment_ratio'], 
                child_pressure, 
                elderly_pressure
            )
            st.text_area('AI-Generated Policy Recommendation', ai_rec, height=200)
            st.caption('This recommendation is AI-assisted and intended to support administrative decision-making.')
    
    st.divider()
    
    # Why is this district risky?
    st.header('Why is this district risky?')
    why_explanation = generate_why_explanation(row, df)
    st.info(why_explanation)
    
else:
    st.warning('No data available for the selected filters. Please adjust your selections.')

st.divider()

# Transparency Note
st.header('Transparency Note')
st.markdown("""
- **ML Model**: Generates quantitative risk scores based on operational data
- **AI Assistant**: Provides contextual explanations and policy suggestions  
- **Human Oversight**: Final decisions remain with administrative authorities
""")

st.divider()

# Model Reliability & Validation
st.header('Model Reliability & Validation')

mae, rmse, spearman_corr, stability = metrics

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Mean Absolute Error', f"{mae:.4f}")

with col2:
    st.metric('Root Mean Squared Error', f"{rmse:.4f}")

with col3:
    st.metric('Spearman Rank Correlation', f"{spearman_corr:.3f}")

with col4:
    st.metric('Top-20 Risk Stability', f"{stability:.1f}%")

st.info("""
This model predicts a continuous risk score and is evaluated on prediction stability and ranking consistency rather than classification accuracy. Low error metrics combined with high rank correlation indicate reliable identification of relative risk levels, making it suitable for policy decision support where ranking districts by risk is more important than exact score prediction.
""")

st.divider()

# Governance Impact
st.header('Governance Impact')
st.markdown("""
This dashboard serves as a critical decision-support tool for Aadhaar ecosystem governance, enabling:

**Proactive Planning**: Identify emerging stress points before they become service disruptions, allowing preemptive resource allocation.

**Targeted Intervention**: Pinpoint specific districts and states requiring immediate attention, ensuring efficient deployment of biometric vans, additional counters, and enrolment camps.

**Data-Driven Governance**: Transform raw operational data into actionable insights, supporting evidence-based policy decisions that optimize citizen service delivery.

**Reduced Service Disruption**: By anticipating and addressing risk factors early, minimize delays in Aadhaar updates and enrolments, maintaining public trust in the system.

This solution demonstrates how AI and data analytics can enhance government service delivery, making Aadhaar operations more resilient and citizen-centric.
""")

st.divider()

# Download Option
st.header('Download Ranked District Stress Data')
ranked_df = df.groupby('district').agg({
    'service_stress_risk': 'mean',
    'biometric_to_enrolment_ratio': 'mean',
    'child_update_pressure': 'mean',
    'elderly_update_pressure': 'mean'
}).reset_index().sort_values('service_stress_risk', ascending=False)
csv = ranked_df.to_csv(index=False)
st.download_button(
    label='Download CSV',
    data=csv,
    file_name='ranked_district_stress.csv',
    mime='text/csv'
)