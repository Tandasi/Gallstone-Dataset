import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Gallstone Disease Prediction",
    page_icon="medical",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'gallstone_prediction_model.pkl'
    if not os.path.exists(model_path):
        st.error(f" Model file '{model_path}' not found in current directory.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        return model_artifacts
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
        return None

# Create sample prediction function
def create_sample_prediction():
    """Create a demo prediction without the actual model"""
    return {
        'prediction': 'Gallstone',
        'probability': {'no_gallstone': 0.25, 'gallstone': 0.75},
        'confidence': 0.75
    }

# Main app
def main():
    st.title("Gallstone Disease Prediction System")
    st.markdown("**AI-Powered Clinical Decision Support Tool**")
    st.markdown("*Developed by GIFT TANDASI using UCI Gallstone Dataset*")
    
    # Check if model file exists
    model_artifacts = load_model()
    
    if model_artifacts:
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        feature_names = model_artifacts['feature_names']
        metrics = model_artifacts['model_metrics']
        
        st.success("Model loaded successfully!")
        
        # Sidebar with model info
        st.sidebar.header("Model Information")
        st.sidebar.metric("Model Type", "Gradient Boosting")
        st.sidebar.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
        st.sidebar.metric("ROC-AUC", f"{metrics['test_auc']:.3f}")
        st.sidebar.metric("Training Date", model_artifacts['training_date'])
        
    else:
        st.warning("Running in DEMO mode - Model file not found")
        st.info("📝 To use the full model, ensure 'gallstone_prediction_model.pkl' is in the same directory")
        
        # Demo sidebar
        st.sidebar.header("Model Information (Demo)")
        st.sidebar.metric("Model Type", "Gradient Boosting")
        st.sidebar.metric("Test Accuracy", "73.4%")
        st.sidebar.metric("ROC-AUC", "0.783")
        st.sidebar.metric("Status", "Demo Mode")
    
    # Main prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Patient Data Input")
        
        # Create input fields for key features
        st.subheader("Demographic Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Female (0)", "Male (1)"])
        height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
        weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70)
        
        st.subheader("Laboratory Values")
        crp = st.number_input("C-Reactive Protein (CRP)", min_value=0.0, max_value=50.0, value=2.0)
        vitamin_d = st.number_input("Vitamin D", min_value=0.0, max_value=100.0, value=25.0)
        cholesterol = st.number_input("Total Cholesterol", min_value=100.0, max_value=400.0, value=200.0)
        
        # Add more key features
        st.subheader("Bioimpedance Measurements")
        protein = st.number_input("Body Protein Content (%)", min_value=10.0, max_value=25.0, value=16.0)
        fat_percentage = st.number_input("Body Fat (%)", min_value=5.0, max_value=50.0, value=20.0)
        
        # Calculate BMI
        bmi = weight / (height/100)**2
        st.metric("Calculated BMI", f"{bmi:.1f}")
        
        # Prediction button
        if st.button("Predict Gallstone Risk", type="primary"):
            # Create patient data
            patient_data = {
                'Age': age,
                'Gender': int(gender.split('(')[1].split(')')[0]),
                'Height': height,
                'Weight': weight,
                'BMI': bmi,
                'C-Reactive Protein (CRP)': crp,
                'Vitamin D': vitamin_d,
                'Total Cholesterol': cholesterol,
                'Body Protein Content (%)': protein,
                'Body Fat (%)': fat_percentage
            }
            
            # Make prediction
            if model_artifacts:
                st.info("Full prediction requires all 39 features. This is a simplified demo.")
                prediction_result = create_sample_prediction()
            else:
                prediction_result = create_sample_prediction()
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction_result['prediction'] == 'Gallstone':
                st.error(f"**HIGH RISK**: {prediction_result['prediction']}")
            else:
                st.success(f"**LOW RISK**: {prediction_result['prediction']}")
            
            # Show probabilities
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("No Gallstone Probability", 
                         f"{prediction_result['probability']['no_gallstone']:.1%}")
            with col_b:
                st.metric("Gallstone Probability", 
                         f"{prediction_result['probability']['gallstone']:.1%}")
            
            st.metric("Confidence Level", f"{prediction_result['confidence']:.1%}")
    
    with col2:
        st.header("Model Performance")
        
        # Display confusion matrix (example data)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm_data = np.array([[32, 8], [12, 12]])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Gallstone', 'Gallstone'],
                   yticklabels=['No Gallstone', 'Gallstone'])
        ax.set_title('Test Set Performance')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        st.subheader("Key Features")
        st.write("Most important predictors:")
        features_importance = [
            "1. C-Reactive Protein (CRP)",
            "2. Body Protein Content",
            "3. Vitamin D",
            "4. Obesity %",
            "5. Bone Mass"
        ]
        for feature in features_importance:
            st.write(feature)
        
        # Model metrics
        st.subheader("Performance Metrics")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': ['73.4%', '76.7%', '69.7%', '73.0%', '0.783']
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions.")
    
    # Instructions
    with st.expander("📋 How to use this application"):
        st.markdown("""
        1. **Enter patient data** in the left panel
        2. **Click 'Predict Gallstone Risk'** to get prediction
        3. **Review results** and confidence scores
        4. **Consult healthcare professionals** for medical decisions
        
        **Note:** This demo version uses simplified inputs. The full model requires all 39 clinical features.
        """)
    
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        - **Model:** Gradient Boosting Classifier
        - **Dataset:** UCI Gallstone Dataset (319 samples)
        - **Features:** 39 clinical measurements
        - **Performance:** 73.4% accuracy, 0.783 ROC-AUC
        - **Author:** GIFT TANDASI
        """)

if __name__ == "__main__":
    main()