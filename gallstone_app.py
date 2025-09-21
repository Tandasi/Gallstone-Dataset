import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
    except (pickle.UnpicklingError, EOFError, FileNotFoundError, PermissionError) as e:
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

def show_prediction_interface(model_artifacts):
    """Display the prediction interface"""
    st.header("Patient Risk Assessment")

    # Main prediction interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Patient Data Input")

        # Create input fields for key features
        with st.container():
            st.markdown("### Demographic Information")
            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                age = st.number_input("Age", min_value=18, max_value=100, value=45)
                gender = st.selectbox("Gender", ["Female", "Male"])
                height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
                weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70)

            with col_demo2:
                # Comorbidities
                st.markdown("**Comorbidities:**")
                comorbidity = st.selectbox("Any Comorbidity?", ["No (0)", "Yes (1)"])
                cad = st.selectbox("Coronary Artery Disease (CAD)", ["No (0)", "Yes (1)"])
                hypothyroidism = st.selectbox("Hypothyroidism", ["No (0)", "Yes (1)"])
                hyperlipidemia = st.selectbox("Hyperlipidemia", ["No (0)", "Yes (1)"])
                dm = st.selectbox("Diabetes Mellitus (DM)", ["No (0)", "Yes (1)"])

        # Calculate BMI
        bmi = weight / (height/100)**2
        st.metric("**Calculated BMI**", f"{bmi:.1f}")
        st.divider()

        st.markdown("### Laboratory Values")
        st.divider()
        col_lab1, col_lab2 = st.columns(2)
        with col_lab1:
            glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=500.0, value=90.0)
            tc = st.number_input("Total Cholesterol (TC) (mg/dL)", min_value=100.0, max_value=400.0, value=200.0)
            ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50.0, max_value=200.0, value=120.0)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20.0, max_value=100.0, value=50.0)
            triglyceride = st.number_input("Triglyceride (mg/dL)", min_value=50.0, max_value=500.0, value=150.0)

        with col_lab2:
            ast = st.number_input("AST/SGOT (U/L)", min_value=10.0, max_value=200.0, value=25.0)
            alt = st.number_input("ALT/SGPT (U/L)", min_value=10.0, max_value=200.0, value=30.0)
            alp = st.number_input("ALP (U/L)", min_value=40.0, max_value=500.0, value=80.0)
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.0)
            gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=30.0, max_value=150.0, value=90.0)

        st.markdown("### Additional Blood Tests")
        st.divider()
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            crp = st.number_input("C-Reactive Protein (CRP) (mg/L)", min_value=0.0, max_value=50.0, value=2.0)
            hgb = st.number_input("Hemoglobin (HGB) (g/dL)", min_value=10.0, max_value=18.0, value=14.0)
            vitamin_d = st.number_input("Vitamin D (ng/mL)", min_value=0.0, max_value=100.0, value=25.0)

        with col_add2:
            # Bioimpedance measurements (REQUIRED for accurate prediction)
            st.write("**Bioimpedance Measurements (Required for Prediction):**")
            tbw = st.number_input("Total Body Water (TBW) (%)", min_value=40.0, max_value=70.0, value=55.0, help="Required for prediction")
            ecw = st.number_input("Extracellular Water (ECW) (L)", min_value=10.0, max_value=30.0, value=15.0, help="Required for prediction")
            icw = st.number_input("Intracellular Water (ICW) (L)", min_value=15.0, max_value=40.0, value=25.0, help="Required for prediction")
            ect_tbw = st.number_input("Extracellular Fluid/Total Body Water (ECF/TBW)", min_value=0.3, max_value=0.7, value=0.4, help="Required for prediction")
            protein_pct = st.number_input("Body Protein Content (%)", min_value=10.0, max_value=25.0, value=16.0, help="Required for prediction")
            fat_pct = st.number_input("Body Fat (%)", min_value=5.0, max_value=50.0, value=20.0, help="Required for prediction")
            tbfr = st.number_input("Total Body Fat Ratio (TBFR) (%)", min_value=5.0, max_value=50.0, value=20.0, help="Required for prediction")
            lm_pct = st.number_input("Lean Mass (LM) (%)", min_value=40.0, max_value=80.0, value=60.0, help="Required for prediction")
            vfr = st.number_input("Visceral Fat Rating", min_value=1, max_value=30, value=10, help="Required for prediction")
            muscle_mass = st.number_input("Muscle Mass (kg)", min_value=30.0, max_value=100.0, value=55.0, help="Required for prediction")
            bone_mass = st.number_input("Bone Mass (kg)", min_value=1.0, max_value=5.0, value=2.5, help="Required for prediction")
            obesity_pct = st.number_input("Obesity (%)", min_value=0.0, max_value=50.0, value=15.0, help="Required for prediction")
            tfc = st.number_input("Total Fat Content (kg)", min_value=5.0, max_value=50.0, value=15.0, help="Required for prediction")
            vfa = st.number_input("Visceral Fat Area (cm²)", min_value=0.0, max_value=200.0, value=50.0, help="Required for prediction")
            vma = st.number_input("Visceral Muscle Area (VMA) (kg)", min_value=20.0, max_value=80.0, value=40.0, help="Required for prediction")
            hfa = st.number_input("Hepatic Fat Accumulation (HFA)", min_value=0.0, max_value=100.0, value=10.0, help="Required for prediction")

        # Calculate BMI again for consistency
        bmi = weight / (height/100)**2
        st.metric("Calculated BMI", f"{bmi:.1f}")
        st.divider()

        # Prediction button
        if st.button("Predict Gallstone Risk", type="primary", use_container_width=True):
            # Create patient data dictionary with all collected features
            patient_data = {
                'Age': age,
                'Gender': 0 if gender == "Female" else 1,
                'Comorbidity': int(comorbidity.split('(')[1].split(')')[0]),
                'Coronary Artery Disease (CAD)': int(cad.split('(')[1].split(')')[0]),
                'Hypothyroidism': int(hypothyroidism.split('(')[1].split(')')[0]),
                'Hyperlipidemia': int(hyperlipidemia.split('(')[1].split(')')[0]),
                'Diabetes Mellitus (DM)': int(dm.split('(')[1].split(')')[0]),
                'Height': height,
                'Weight': weight,
                'Body Mass Index (BMI)': bmi,
                'Total Body Water (TBW)': tbw,
                'Extracellular Water (ECW)': ecw,
                'Intracellular Water (ICW)': icw,
                'Extracellular Fluid/Total Body Water (ECF/TBW)': ect_tbw,
                'Total Body Fat Ratio (TBFR) (%)': tbfr,
                'Lean Mass (LM) (%)': lm_pct,
                'Body Protein Content (Protein) (%)': protein_pct,
                'Visceral Fat Rating (VFR)': vfr,
                'Bone Mass (BM)': bone_mass,
                'Muscle Mass (MM)': muscle_mass,
                'Obesity (%)': obesity_pct,
                'Total Fat Content (TFC)': tfc,
                'Visceral Fat Area (VFA)': vfa,
                'Visceral Muscle Area (VMA) (Kg)': vma,
                'Hepatic Fat Accumulation (HFA)': hfa,
                'Body Fat (%)': fat_pct,
                'Glucose': glucose,
                'Total Cholesterol (TC)': tc,
                'Low Density Lipoprotein (LDL)': ldl,
                'High Density Lipoprotein (HDL)': hdl,
                'Triglyceride': triglyceride,
                'Aspartat Aminotransferaz (AST)': ast,
                'Alanin Aminotransferaz (ALT)': alt,
                'Alkaline Phosphatase (ALP)': alp,
                'Creatinine': creatinine,
                'Glomerular Filtration Rate (GFR)': gfr,
                'C-Reactive Protein (CRP)': crp,
                'Hemoglobin (HGB)': hgb,
                'Vitamin D': vitamin_d,
                'BMI': bmi  # Duplicate for compatibility
            }

            # Make prediction
            if model_artifacts and len(patient_data) >= 39:  # All 39 features required
                try:
                    # Assign model, scaler, and feature_names here to ensure they are defined
                    model = model_artifacts['model']
                    scaler = model_artifacts['scaler']
                    feature_names = model_artifacts['feature_names']

                    # Convert to DataFrame
                    patient_df = pd.DataFrame([patient_data])

                    # Check if all required features are present
                    missing_features = [f for f in feature_names if f not in patient_df.columns]
                    if missing_features:
                        st.error(f"Missing required features: {', '.join(missing_features)}")
                        prediction_result = create_sample_prediction()
                        st.info("Using demo prediction due to missing features")
                    else:
                        # All features present, proceed with prediction
                        X_patient = patient_df[feature_names]  # Use all features in correct order
                        X_patient_scaled = scaler.transform(X_patient)

                        # Make prediction
                        prediction_proba = model.predict_proba(X_patient_scaled)[0]
                        prediction = model.predict(X_patient_scaled)[0]

                        prediction_result = {
                            'prediction': 'Gallstone' if prediction == 1 else 'No Gallstone',
                            'probability': {
                                'no_gallstone': prediction_proba[0],
                                'gallstone': prediction_proba[1]
                            },
                            'confidence': max(prediction_proba)
                        }
                        st.success("Prediction completed using trained model!")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    prediction_result = create_sample_prediction()
                    st.info("Using demo prediction due to processing error")
            else:
                prediction_result = create_sample_prediction()
                st.info("Using demo prediction (all 39 features required for accurate prediction)")
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
        st.subheader("Model Performance")
        st.divider()

        # Display confusion matrix (example data)
        with st.container():
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(6, 4))
            cm_data = np.array([[32, 8], [12, 12]])
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Gallstone', 'Gallstone'],
                       yticklabels=['No Gallstone', 'Gallstone'])
            ax.set_title('Test Set Performance')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

        st.markdown("**Key Features**")
        st.write("Most important predictors:")
        features_importance = [
            "1. C-Reactive Protein (CRP)",
            "2. Body Protein Content",
            "3. Vitamin D",
            "4. Obesity %",
            "5. Bone Mass"
        ]
        for feature in features_importance:
            st.write(f"• {feature}")

        # Model metrics
        st.markdown("**Performance Metrics**")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': ['73.4%', '76.7%', '69.7%', '73.0%', '0.783']
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, width='stretch')

def show_dataset_analysis():
    """Display dataset analysis section"""
    st.header("Dataset Analysis & Exploration")

    try:
        # Load and cache dataset
        @st.cache_data
        def load_dataset():
            return pd.read_csv('dataset-uci.csv')
        
        df = load_dataset()
        st.markdown(f"**Dataset Overview:** {df.shape[0]} patients × {df.shape[1]} features")

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.markdown("### Gender Distribution")
            gender_counts = df['Gender'].value_counts()
            females = gender_counts.get(0, 0)
            males = gender_counts.get(1, 0)
            st.write(f"**Females:** {females} patients ({females/len(df)*100:.1f}%)")
            st.write(f"**Males:** {males} patients ({males/len(df)*100:.1f}%)")

            st.markdown("### Target Distribution")
            target_counts = df.iloc[:, -1].value_counts()  # Assuming target is last column
            no_gallstone = target_counts.get(0, 0)
            gallstone = target_counts.get(1, 0)
            st.write(f"**No Gallstone:** {no_gallstone} patients ({no_gallstone/len(df)*100:.1f}%)")
            st.write(f"**Gallstone:** {gallstone} patients ({gallstone/len(df)*100:.1f}%)")

        with col_info2:
            st.markdown("### Key Clinical Features")
            features = [
                "Age, Gender, Height, Weight, BMI",
                "Comorbidities (CAD, DM, etc.)",
                "Lipid Profile (TC, LDL, HDL, TG)",
                "Liver Enzymes (AST, ALT, ALP)",
                "Kidney Function (Creatinine, GFR)",
                "Inflammation (CRP, Hemoglobin)",
                "Bioimpedance Measurements",
                "Vitamin D levels"
            ]
            for feature in features:
                st.write(f"• {feature}")

        st.markdown("### Sample Data (First 10 rows)")
        st.dataframe(df.head(10), width='stretch')

        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), width='stretch')

        # Add visualizations
        st.markdown("---")
        st.markdown("## Dataset Visualizations")

        # Create tabs for different types of visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Distributions", "Relationships", "Correlations"])

        with viz_tab1:
            st.markdown("### Target Variable Distribution")
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                # Pie chart for target distribution
                fig, ax = plt.subplots(figsize=(5, 5))  # Reduced size for cloud compatibility
                target_counts = df['Gallstone Status'].value_counts()
                labels = ['No Gallstone', 'Gallstone']
                sizes = [target_counts.get(0, 0), target_counts.get(1, 0)]
                colors = ['#66b3ff', '#ff9999']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title('Gallstone Status Distribution')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            with col_viz2:
                # Age distribution histogram
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.histplot(data=df, x='Age', hue='Gallstone Status', multiple="stack", ax=ax, bins=20)
                ax.set_title('Age Distribution by Gallstone Status')
                ax.set_xlabel('Age (years)')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            # Gender distribution
            st.markdown("### Gender Distribution")
            gender_counts = df['Gender'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
            gender_labels = ['Female', 'Male']
            gender_sizes = [gender_counts.get(0, 0), gender_counts.get(1, 0)]
            ax.bar(gender_labels, gender_sizes, color=['#ff9999', '#66b3ff'])
            ax.set_title('Gender Distribution')
            ax.set_ylabel('Count')
            for i, v in enumerate(gender_sizes):
                ax.text(i, v + 1, str(v), ha='center')
            st.pyplot(fig)
            plt.close(fig)  # Free memory

            # BMI Distribution
            st.markdown("### BMI Distribution Analysis")
            col_bmi1, col_bmi2 = st.columns(2)

            with col_bmi1:
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.histplot(data=df, x='Body Mass Index (BMI)', hue='Gallstone Status', multiple="stack", ax=ax, bins=15)
                ax.set_title('BMI Distribution by Gallstone Status')
                ax.set_xlabel('BMI')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            with col_bmi2:
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.boxplot(data=df, x='Gallstone Status', y='Body Mass Index (BMI)', ax=ax)
                ax.set_title('BMI Box Plot by Gallstone Status')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['No Gallstone', 'Gallstone'])
                ax.set_xlabel('Gallstone Status')
                ax.set_ylabel('BMI')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            # Key Laboratory Values Distribution (simplified for cloud)
            st.markdown("### Key Laboratory Values")
            lab_cols = ['Glucose', 'Total Cholesterol (TC)', 'Triglyceride', 'C-Reactive Protein (CRP)']

            for i in range(0, len(lab_cols), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(lab_cols):
                        feature = lab_cols[i + j]
                        with cols[j]:
                            try:
                                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                                sns.histplot(data=df, x=feature, hue='Gallstone Status', multiple="stack", ax=ax, bins=15)
                                ax.set_title(f'{feature} Distribution')
                                ax.set_xlabel(feature)
                                ax.set_ylabel('Count')
                                st.pyplot(fig)
                                plt.close(fig)  # Free memory
                            except Exception as plot_error:
                                st.warning(f"Could not create plot for {feature}: {str(plot_error)}")

            # Bioimpedance Features (simplified for cloud)
            st.markdown("### Key Bioimpedance Features")
            bio_cols = ['Total Body Water (TBW)', 'Lean Mass (LM) (%)', 'Total Body Fat Ratio (TBFR) (%)']

            for i in range(0, len(bio_cols), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(bio_cols):
                        feature = bio_cols[i + j]
                        with cols[j]:
                            try:
                                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                                sns.histplot(data=df, x=feature, hue='Gallstone Status', multiple="stack", ax=ax, bins=15)
                                ax.set_title(f'{feature} Distribution')
                                ax.set_xlabel(feature)
                                ax.set_ylabel('Count')
                                st.pyplot(fig)
                                plt.close(fig)  # Free memory
                            except Exception as plot_error:
                                st.warning(f"Could not create plot for {feature}: {str(plot_error)}")

        with viz_tab2:
            st.markdown("### Key Feature Relationships")

            col_rel1, col_rel2 = st.columns(2)

            with col_rel1:
                # BMI vs Age scatter plot
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.scatterplot(data=df, x='Age', y='Body Mass Index (BMI)', hue='Gallstone Status', ax=ax, alpha=0.6)
                ax.set_title('BMI vs Age by Gallstone Status')
                ax.set_xlabel('Age (years)')
                ax.set_ylabel('BMI')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            with col_rel2:
                # CRP vs BMI
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.scatterplot(data=df, x='Body Mass Index (BMI)', y='C-Reactive Protein (CRP)', hue='Gallstone Status', ax=ax, alpha=0.6)
                ax.set_title('CRP vs BMI by Gallstone Status')
                ax.set_xlabel('BMI')
                ax.set_ylabel('CRP (mg/L)')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            # Metabolic relationships
            st.markdown("### Metabolic Indicators")
            col_met1, col_met2 = st.columns(2)

            with col_met1:
                # Glucose vs BMI
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.scatterplot(data=df, x='Body Mass Index (BMI)', y='Glucose', hue='Gallstone Status', ax=ax, alpha=0.6)
                ax.set_title('Glucose vs BMI by Gallstone Status')
                ax.set_xlabel('BMI')
                ax.set_ylabel('Glucose (mg/dL)')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

            with col_met2:
                # Triglyceride vs HDL
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.scatterplot(data=df, x='High Density Lipoprotein (HDL)', y='Triglyceride', hue='Gallstone Status', ax=ax, alpha=0.6)
                ax.set_title('Triglyceride vs HDL by Gallstone Status')
                ax.set_xlabel('HDL (mg/dL)')
                ax.set_ylabel('Triglyceride (mg/dL)')
                st.pyplot(fig)
                plt.close(fig)  # Free memory

        with viz_tab3:
            st.markdown("### Feature Correlations")

            # Select numeric columns for correlation
            numeric_cols = ['Age', 'Height', 'Weight', 'Body Mass Index (BMI)', 'Glucose',
                          'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
                          'High Density Lipoprotein (HDL)', 'Triglyceride', 'Vitamin D',
                          'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)']

            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Correlation Matrix of Key Clinical Features')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)

            # Top correlations with target
            st.markdown("### Feature Importance (Correlation with Gallstone Status)")
            target_corr = df[numeric_cols + ['Gallstone Status']].corr()['Gallstone Status'].abs().sort_values(ascending=False)
            target_corr = target_corr.drop('Gallstone Status')  # Remove self-correlation

            fig, ax = plt.subplots(figsize=(10, 6))
            target_corr.head(10).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title('Top 10 Features Correlated with Gallstone Status')
            ax.set_xlabel('Absolute Correlation')
            st.pyplot(fig)
            plt.close(fig)  # Free memory

            # Simplified correlation matrix
            st.markdown("### Key Clinical Features Correlation")
            fig, ax = plt.subplots(figsize=(8, 6))  # Reduced size
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, ax=ax)  # Removed annotations for cloud compatibility
            ax.set_title('Clinical Features Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            plt.close(fig)  # Free memory

    except Exception as e:
        st.error(f"Could not load dataset: {str(e)}")
        st.info("Make sure 'dataset-uci.csv' is in the same directory as the app.")

def show_about_section():
    """Display about section"""
    st.header("About This Application")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### How to Use")
        st.markdown("""
        1. **Navigate to Prediction tab**
        2. **Enter patient clinical data**
        3. **Click 'Predict Gallstone Risk'**
        4. **Review results and confidence scores**
        5. **Consult healthcare professionals**

        **Note:** This tool is for research and educational purposes only.
        """)

    with col2:
        st.markdown("### Technical Details")
        st.markdown("""
        - **Model:** Gradient Boosting Classifier
        - **Dataset:** UCI Gallstone Dataset
        - **Samples:** 319 patients
        - **Features:** 39 clinical measurements
        - **Performance:** 73.4% accuracy, 0.783 ROC-AUC
        - **Author:** GIFT TANDASI
        - **Framework:** Streamlit + scikit-learn
        """)

    st.markdown("---")
    st.markdown("### Important Disclaimer")
    st.warning("""
    **This application is for educational and research purposes only.**
    Always consult qualified healthcare professionals for medical decisions.
    The predictions provided are not a substitute for professional medical advice,
    diagnosis, or treatment.
    """)

# Main app
def main():
    st.title("Gallstone Disease Prediction System")
    st.markdown("**AI-Powered Clinical Decision Support Tool**")
    st.markdown("*Developed by GIFT TANDASI using UCI Gallstone Dataset*")
    st.markdown("---")

    # Check if model file exists
    model_artifacts = load_model()

    if model_artifacts:
        metrics = model_artifacts['model_metrics']

        st.success("Model loaded successfully!")

        # Sidebar with model info
        with st.sidebar:
            st.header("Model Information")
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("**Accuracy**", f"{metrics['test_accuracy']:.1%}")
                st.metric("**ROC-AUC**", f"{metrics['test_auc']:.3f}")
            with col2:
                st.metric("**Samples**", "319")
                st.metric("**Features**", "39")

            st.markdown("### Model Details")
            st.info(f"""
            • **Type:** Gradient Boosting  
            • **Training Date:** {model_artifacts.get('training_date', 'Unknown')}  
            • **Status:** Active
            """)

    else:
        st.warning("Running in DEMO mode - Model file not found")
        st.info("To use the full model, ensure 'gallstone_prediction_model.pkl' is in the same directory")

        # Demo sidebar
        with st.sidebar:
            st.header("Model Information (Demo)")
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("**Accuracy**", "73.4%")
                st.metric("**ROC-AUC**", "0.783")
            with col2:
                st.metric("**Samples**", "319")
                st.metric("**Features**", "39")

            st.markdown("### Model Details")
            st.warning("""
            • **Type:** Gradient Boosting  
            • **Status:** Demo Mode
            """)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Analysis", "About"])

    with tab1:
        show_prediction_interface(model_artifacts)

    with tab2:
        show_dataset_analysis()

    with tab3:
        show_about_section()

if __name__ == "__main__":
    main()