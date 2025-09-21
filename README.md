# Gallstone Disease Prediction Model

**AI-Powered Clinical Decision Support System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project implements a machine learning model to predict gallstone disease using demographic, bioimpedance, and laboratory data. The model achieves **73.4% accuracy** with explainable AI features using SHAP analysis.

**Author:** GIFT TANDASI  
**Dataset:** UCI Gallstone Dataset (319 samples, 39 features)  
**Repository:** https://github.com/Tandasi/Gallstone-Dataset  

## Key Features

- **High Accuracy:** 73.4% test accuracy with 0.783 ROC-AUC
- **Explainable AI:** SHAP analysis for transparent predictions
- **Multiple Deployment Options:** Streamlit, Flask API, Docker
- **Clinical Ready:** Designed for healthcare integration
- **Complete Pipeline:** From data loading to deployment

## Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 73.4% |
| **Precision** | 76.7% |
| **Recall** | 69.7% |
| **F1-Score** | 73.0% |
| **ROC-AUC** | 0.783 |

## Dataset Information

- **Source:** UCI Machine Learning Repository
- **Samples:** 319 patients
- **Features:** 39 (demographic, bioimpedance, laboratory)
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
streamlit run streamlit_app.py
```

### 4. Run API Server
```bash
python flask_app.py
```

## Project Structure

```
gallstone-1/
├── UCI.ipynb                           # Main analysis notebook
├── dataset-uci.csv                     # Dataset file
├── gallstone_prediction_model.pkl      # Trained model
├── streamlit_app.py                    # Web application
├── flask_app.py                        # REST API
├── requirements.txt                    # Dependencies
├── Dockerfile                          # Container setup
├── docker-compose.yml                  # Multi-container setup
├── Procfile                            # Heroku deployment
└── README.md                           # This file
```

## Deployment Options

### Option 1: Streamlit Cloud (Recommended for Demo)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy
4. **Pros:** Free, easy, no maintenance
5. **Best for:** Research, demos, portfolios

### Option 2: Heroku (Quick Production)
```bash
heroku create your-app-name
git push heroku main
```
**Pros:** Fast deployment, good for prototypes  
**Best for:** MVPs, small-scale applications

### Option 3: Docker (Scalable)
```bash
docker build -t gallstone-api .
docker run -p 5000:5000 gallstone-api
```
**Pros:** Consistent environment, scalable  
**Best for:** Production, cloud deployment

### Option 4: AWS/Azure (Enterprise)
- Use SageMaker, Azure ML Studio
- HIPAA-compliant for healthcare
- **Best for:** Clinical integration, large scale

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
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.1
```

### Web Framework
```txt
streamlit==1.25.0
flask==2.3.3
gunicorn==21.2.0
```

### Development
```txt
jupyter==1.0.0
notebook==6.5.4
```

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Model Validation
```bash
python validate_model.py
```

### API Testing
```bash
python test_api.py
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
**Email:** [your.email@example.com]  
**LinkedIn:** [Your LinkedIn Profile]  
**GitHub:** [Your GitHub Profile]  

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

---

**Built with love for advancing healthcare through AI**