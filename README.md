# Gallstone Disease Prediction Model

**AI-Powered Clinical Decision Support System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project implements a machine learning model to predict gallstone disease using demographic, bioimpedance, and laboratory data. The model achieves **73.4% accuracy** with explainable AI features using SHAP analysis.

**Author:** GIFT TANDASI  
**Dataset:** UCI Gallstone Dataset (319 samples, 38 features)  
**Repository:** https://github.com/Tandasi/Gallstone-Dataset  

## Key Features

- **High Accuracy:** 73.4% test accuracy with 0.783 ROC-AUC
- **Explainable AI:** SHAP analysis for transparent predictions
- **Web Deployment:** Streamlit Cloud deployment ready
- **Clinical Ready:** Designed for healthcare integration
- **Complete Pipeline:** From data loading to deployment

## Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 73.4% |
| **Precision** | 81.8% |
| **Recall** | 58.1% |
| **F1-Score** | 67.9% |
| **ROC-AUC** | 0.783 |

## Dataset Information

- **Source:** UCI Machine Learning Repository
- **Samples:** 319 patients
- **Features:** 38 input features (demographic, bioimpedance, laboratory)
- **Target:** Binary classification (Gallstone vs No Gallstone)
- **Balance:** 50.5% no gallstone, 49.5% gallstone
- **Missing Values:** None

### Key Features
- **Demographic:** Age, gender, height, weight, BMI
- **Bioimpedance:** Body composition, muscle mass, fat percentage
- **Laboratory:** CRP, cholesterol, vitamins, liver enzymes

## Quick Start

### Prerequisites
```bash
Python 3.9+
pip install -r requirements.txt
```

### 1. Clone and Setup
```bash
git clone https://github.com/Tandasi/Gallstone-Dataset.git
cd gallstone-1
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```bash
jupyter notebook UCI.ipynb
```

### 3. Run Web Application
```bash
streamlit run gallstone_app.py
```

## Project Structure

```
gallstone-1/
├── UCI.ipynb                           # Main analysis notebook
├── dataset-uci.csv                     # Dataset file
├── gallstone_prediction_model.pkl      # Trained model
├── gallstone_app.py                    # Streamlit web application
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```


## API Usage

### Prediction Endpoint
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "Age": 45,
      "Gender": 1,
      "Height": 170,
      "Weight": 70,
      "C-Reactive Protein (CRP)": 2.5,
      "Vitamin D": 25.0,
      ...
    }
  }'
```

### Response Format
```json
{
  "prediction": "Gallstone",
  "probability": {
    "no_gallstone": 0.25,
    "gallstone": 0.75
  },
  "confidence": 0.75,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Model Details

### Algorithm: Gradient Boosting Classifier
- **Best performing** among 9 tested algorithms
- **Cross-validation:** 5-fold CV for robust evaluation
- **Feature importance:** Available via SHAP analysis

### Top Predictive Features:
1. **C-Reactive Protein (CRP)** - Inflammation marker
2. **Body Protein Content** - Bioimpedance measurement
3. **Vitamin D** - Laboratory value
4. **Obesity Percentage** - Body composition
5. **Bone Mass** - Bioimpedance measurement

## Model Interpretability

### SHAP Analysis
- **Summary plots:** Feature importance visualization
- **Waterfall plots:** Individual prediction explanations
- **Force plots:** Decision boundary analysis

### Example Interpretation
"Patient has **75% probability** of gallstone due to:
- High CRP levels (+0.15)
- Low Vitamin D (-0.08)
- High obesity percentage (+0.12)"

## Healthcare Compliance

### Security Features
- Input validation and sanitization
- Error handling and logging
- Secure API endpoints
- Data anonymization support

### HIPAA Considerations
- No PHI storage in model
- Encrypted data transmission
- Audit trail capabilities
- Access control ready

## Installation & Dependencies

### Core Dependencies
```txt
streamlit>=1.28.0
pandas>=1.5.3
numpy>=1.24.3
matplotlib>=3.8.0
seaborn>=0.12.2
shap>=0.42.1
scikit-learn>=1.3.0
pillow>=10.0.0
```

### Development
```txt
jupyter==1.0.0
notebook==6.5.4
```

## Testing

### Basic Functionality Test
```bash
# Test that the Streamlit app starts without errors
streamlit run gallstone_app.py --server.headless true --server.port 8501
```

### Model Loading Test
```bash
python -c "import pickle; model = pickle.load(open('gallstone_prediction_model.pkl', 'rb')); print('Model loaded successfully')"
```

### Data Loading Test
```bash
python -c "import pandas as pd; df = pd.read_csv('dataset-uci.csv'); print(f'Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features')"
```

## Performance Monitoring

### Key Metrics to Track
- **Prediction accuracy** over time
- **Data drift** detection
- **Response time** monitoring
- **Error rates** tracking
- **Usage patterns** analysis

### Recommended Tools
- **Monitoring:** Prometheus, Grafana
- **Logging:** ELK Stack, CloudWatch
- **Alerting:** PagerDuty, Slack

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Support

**Author:** GIFT TANDASI  
**Email:** gift@anga-tech.com  
**GitHub:** https://github.com/Tandasi  

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- Ankara VM Medical Park Hospital for data collection
- Open-source community for tools and libraries
- Healthcare professionals for domain expertise

## Roadmap

### Version 2.0 Features
- [ ] Enhanced feature engineering
- [ ] Ensemble model improvements
- [ ] Real-time monitoring dashboard
- [ ] Mobile application
- [ ] Integration with EHR systems

### Future Enhancements
- [ ] Multi-language support
- [ ] Advanced visualization dashboard
- [ ] Batch prediction capabilities
- [ ] Model retraining automation
- [ ] Clinical trial integration

---

## Disclaimer

This tool is for **research and educational purposes only**. Always consult qualified healthcare professionals for medical decisions. The model should not be used as the sole basis for clinical diagnosis or treatment decisions.

