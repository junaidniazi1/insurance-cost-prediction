import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animated gradient title */
    .hero-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ffffff, #ffd700, #ffffff, #87ceeb);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 3rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.2);
    }
    
    /* Premium result card */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(25px);
        border-radius: 32px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        padding: 3rem;
        text-align: center;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: rotate 4s linear infinite;
        z-index: -1;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Floating animation for cards */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .float-animation {
        animation: float 6s ease-in-out infinite;
    }
    
    /* Enhanced metrics */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffd700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Form styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Pulse animation for prediction button */
    .predict-button {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load Gradient Boosting model and scaler"""
    try:
        model = joblib.load("gradient_boosting_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.info(f"🤖 Using demo model for demonstration purposes")
        return create_demo_model()


def create_demo_model():
    """Enhanced demo model with realistic coefficients"""
    demo_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    demo_scaler = StandardScaler()
    
    # Create more realistic demo data
    np.random.seed(42)
    n_samples = 1000
    ages = np.random.randint(18, 65, n_samples)
    sexes = np.random.randint(0, 2, n_samples)
    bmis = np.random.normal(26, 4, n_samples)
    children = np.random.poisson(1.2, n_samples)
    smokers = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    regions = np.random.randint(0, 4, n_samples)
    
    # Create realistic premium calculation
    base_premium = 5000
    age_factor = ages * 50
    bmi_penalty = np.maximum(0, bmis - 25) * 100
    smoker_penalty = smokers * 15000
    children_cost = children * 1500
    premiums = base_premium + age_factor + bmi_penalty + smoker_penalty + children_cost + np.random.normal(0, 1000, n_samples)
    premiums = np.maximum(1000, premiums)  # Minimum premium
    
    X = np.column_stack([ages, sexes, bmis, children, smokers, regions])
    demo_scaler.fit(X)
    demo_model.fit(demo_scaler.transform(X), premiums)
    
    return demo_model, demo_scaler


def create_premium_visualization(prediction):
    """Enhanced gauge chart with beautiful styling"""
    # Determine color based on premium amount
    if prediction < 10000:
        color = "#4CAF50"  # Green
        risk_level = "Low Premium"
    elif prediction < 25000:
        color = "#FF9800"  # Orange
        risk_level = "Medium Premium"
    else:
        color = "#F44336"  # Red
        risk_level = "High Premium"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<span style='color: white; font-size: 24px; font-weight: 600;'>Annual Premium</span><br><span style='color: {color}; font-size: 16px;'>{risk_level}</span>",
            'font': {'color': 'white'}
        },
        number={'font': {'color': 'white', 'size': 48}},
        gauge={
            'axis': {
                'range': [None, 50000],
                'tickwidth': 1,
                'tickcolor': "rgba(255, 255, 255, 0.7)",
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.3)",
            'steps': [
                {'range': [0, 10000], 'color': "rgba(76, 175, 80, 0.2)"},
                {'range': [10000, 25000], 'color': "rgba(255, 152, 0, 0.2)"},
                {'range': [25000, 50000], 'color': "rgba(244, 67, 54, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        height=400
    )
    
    return fig


def display_premium_breakdown(prediction, age, smoker, bmi, children):
    """Display premium breakdown with beautiful metrics"""
    base_premium = prediction * 0.4
    age_factor = (age - 18) * 100
    smoker_factor = prediction * 0.3 if smoker == "Yes" else 0
    bmi_factor = max(0, (float(bmi) - 25) * 200)
    children_factor = children * 1000
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">${prediction/12:,.0f}</div>
            <div class="metric-label">Monthly Premium</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">${prediction/365:.2f}</div>
            <div class="metric-label">Daily Cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">${prediction:,.0f}</div>
            <div class="metric-label">Annual Total</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Hero section
    st.markdown('<h1 class="hero-title">🏥 Smart Insurance Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-powered premium prediction with beautiful insights</p>', unsafe_allow_html=True)

    # Load models
    model, scaler = load_models()

    # Main content
    col1, col2 = st.columns([1.2, 1.8], gap="large")

    with col1:
        st.markdown('<div class="glass-card float-animation">', unsafe_allow_html=True)
        st.markdown("### 📊 Personal Information")
        
        # Age input with custom styling
        age = st.slider("🎂 Age", 18, 100, 30, help="Your current age")
        
        # Gender selection
        sex = st.selectbox("⚧ Gender", ["Female", "Male"], help="Biological gender for insurance purposes")
        sex_encoded = 0 if sex == "Female" else 1
        
        # BMI input
        bmi = st.slider("⚖️ BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1, help="Your BMI calculated as weight(kg)/height(m)²")
        
        # BMI interpretation
        if bmi < 18.5:
            bmi_status = "Underweight 📉"
            bmi_color = "#2196F3"
        elif bmi < 25:
            bmi_status = "Normal 💚"
            bmi_color = "#4CAF50"
        elif bmi < 30:
            bmi_status = "Overweight 🟡"
            bmi_color = "#FF9800"
        else:
            bmi_status = "Obese 🔴"
            bmi_color = "#F44336"
        
        st.markdown(f'<p style="color: {bmi_color}; font-weight: 600; text-align: center; margin-top: -10px;">BMI Status: {bmi_status}</p>', unsafe_allow_html=True)
        
        # Children count
        children = st.slider("👶 Number of Children", 0, 10, 0, help="Number of dependents covered by insurance")
        
        # Smoking status
        smoker = st.selectbox("🚬 Smoking Status", ["No", "Yes"], help="Current smoking status significantly affects premiums")
        smoker_encoded = 0 if smoker == "No" else 1
        
        if smoker == "Yes":
            st.warning("⚠️ Smoking significantly increases premium costs")
        
        # Region selection
        region = st.selectbox("📍 Region", ["Southwest", "Southeast", "Northwest", "Northeast"], help="Your geographical region")
        region_mapping = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
        region_encoded = region_mapping[region]
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Premium Prediction")
        
        # Prediction button with enhanced styling
        if st.button("✨ Calculate Premium", key="predict_btn"):
            # Show loading animation
            with st.spinner("🤖 AI is analyzing your profile..."):
                # Prepare input data
                input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
                
                # Make prediction
                if scaler is not None:
                    input_scaled = scaler.transform(input_data)
                    prediction = max(1000, model.predict(input_scaled)[0])  # Minimum premium of $1000
                else:
                    prediction = max(1000, model.predict(input_data)[0])
                
                # Store prediction in session state
                st.session_state.prediction = prediction
                st.session_state.predicted = True
        
        # Display results if prediction exists
        if hasattr(st.session_state, 'predicted') and st.session_state.predicted:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # Create and display gauge
            fig = create_premium_visualization(st.session_state.prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display premium breakdown
            display_premium_breakdown(st.session_state.prediction, age, smoker, bmi, children)
            
            # Risk assessment
            if st.session_state.prediction < 10000:
                risk_emoji = "💚"
                risk_text = "Low Risk Profile"
                risk_description = "Great! You have a low-risk profile with affordable premiums."
            elif st.session_state.prediction < 25000:
                risk_emoji = "🟡"
                risk_text = "Medium Risk Profile"
                risk_description = "You have a moderate risk profile. Consider healthy lifestyle changes."
            else:
                risk_emoji = "🔴"
                risk_text = "High Risk Profile"
                risk_description = "High-risk profile detected. Consult with healthcare providers about risk reduction."
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 16px;">
                <h3 style="margin: 0; color: white;">{risk_emoji} {risk_text}</h3>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">{risk_description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Information section
    st.markdown('<div class="glass-card" style="margin-top: 3rem;">', unsafe_allow_html=True)
    
    # Create tabs for information
    tab1, tab2, tab3 = st.tabs(["📈 How It Works", "🧠 AI Model Info", "⚠️ Disclaimer"])
    
    with tab1:
        st.markdown("""
        ### 🤖 Machine Learning Prediction
        Our AI model analyzes multiple factors to predict your insurance premium:
        
        **Key Factors:**
        - **Age**: Older individuals typically have higher premiums
        - **Smoking Status**: Smokers face significantly higher costs
        - **BMI**: Higher BMI can indicate health risks
        - **Children**: More dependents increase coverage costs
        - **Region**: Different areas have varying healthcare costs
        - **Gender**: Statistical differences in healthcare utilization
        
        The model uses advanced **Gradient Boosting** algorithms to provide accurate predictions based on historical insurance data.
        """)
    
    with tab2:
        st.markdown("""
        ### 🧠 Technical Details
        **Model Type**: Gradient Boosting Regressor
        **Features**: Age, Gender, BMI, Children, Smoking Status, Region
        **Training**: Trained on comprehensive insurance datasets
        **Accuracy**: Optimized for real-world premium prediction
        
        **Feature Importance:**
        1. 🚬 Smoking Status (Highest impact)
        2. 🎂 Age
        3. ⚖️ BMI
        4. 👶 Number of Children
        5. 📍 Geographic Region
        6. ⚧ Gender
        """)
    
    with tab3:
        st.markdown("""
        ### ⚠️ Important Notice
        **This is a demonstration application** designed to showcase AI-powered insurance premium prediction.
        
        **Please Note:**
        - Predictions are estimates based on historical data patterns
        - Actual premiums may vary significantly between insurance providers
        - This tool should not be used for actual insurance decisions
        - Consult with licensed insurance agents for accurate quotes
        - Individual health conditions and other factors may significantly impact actual premiums
        
        **For Educational Purposes Only** 📚
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()