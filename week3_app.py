import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Telco Churn ML Scoring Engine", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('reports/churn_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/cleaned_telco_churn.csv')

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

st.title("🔮 Predictive Churn Modeling & Risk Simulator")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Architecture", "Logistic Regression")
col2.metric("ROC-AUC Score", "0.8419")
col3.metric("Churn Recall Rate", "78.88%")
col4.metric("At-Risk MRR", f"${df[df['Churn']=='Yes']['MonthlyCharges'].sum():,.2f}")

st.divider()

st.sidebar.header("🎛️ Customer Risk Simulator")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 6)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 85.0)
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🎯 Prediction Result")
    
    # Exact matching values and data types matching cleaned_telco_churn.csv
    sample_df = pd.DataFrame([{
        'gender': 'Male',
        'SeniorCitizen': str(senior),
        'Partner': str(partner),
        'Dependents': str(dependents),
        'tenure': int(tenure),
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': str(internet),
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': str(tech_support),
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': str(contract),
        'PaperlessBilling': str(paperless),
        'PaymentMethod': str(payment_method),
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(monthly_charges * tenure)
    }])
    
    if st.button("⚡ Predict Churn Probability"):
        prob = model.predict_proba(sample_df)[0][1]
        risk_percentage = round(prob * 100, 2)
        
        st.markdown(f"### Predicted Churn Risk: **{risk_percentage}%**")
        st.progress(prob)
        
        if prob >= 0.50:
            st.error("⚠️ **HIGH CHURN RISK DETECTED**\n\nRecommended Retention Action: Offer a 1-Year contract discount or complimentary TechSupport.")
        else:
            st.success("✅ **LOW CHURN RISK**\n\nRecommended Action: Maintain standard onboarding communication.")

with right_col:
    st.subheader("📊 High-Value At-Risk Accounts")
    high_risk_batch = df[(df['Contract'] == 'Month-to-month') & (df['MonthlyCharges'] > 70)].head(10)
    st.dataframe(high_risk_batch[['customerID', 'tenure', 'Contract', 'PaymentMethod', 'MonthlyCharges']], hide_index=True)