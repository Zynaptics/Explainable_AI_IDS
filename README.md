# Explainable AI Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-lightgrey)

A comprehensive AI-powered Intrusion Detection System (IDS) that combines multiple machine learning models with explainable AI techniques for transparent and interpretable network threat detection.

## 🚀 Features

### 🔍 Multi-Model Detection
- **XGBoost Ensemble**: High-performance gradient boosting for accurate threat classification
- **Baseline XGBoost**: Reference model for performance comparison
- **Attention-based Neural Network**: Deep learning with interpretable attention mechanisms
- **Ensemble Voting**: Combined predictions from all models for robust detection

### 📊 Explainable AI (XAI)
- **SHAP Explanations**: Global and local feature importance analysis
- **Attention Weights**: Visualize what the neural network focuses on
- **Model Comparisons**: Side-by-side performance and prediction analysis
- **Feature Impact**: Understand which factors drive detection decisions

### 🎯 Professional Dashboard
- **Real-time Threat Assessment**: Live network traffic analysis
- **Interactive Visualizations**: Dynamic charts and explainability plots
- **Multi-page Interface**: Organized navigation for different analysis types
- **Professional UI**: Clean, enterprise-ready interface


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

###  Clone the Repository
bash
git clone https://github.com/zynaptics/Explainable_AI_IDS.git
cd Explainable_AI_IDS

### Install Dependencies
bash
pip install -r requirements.txt

### Launch the Dashboard
bash
cd dashboard
streamlit run app.py
The dashboard will open automatically in your default browser at http://localhost:8501


### Dashboard Overview
System metrics and performance indicators

Sample network traffic data preview

Feature distribution analysis

Model loading status

### Feature Analysis
SHAP Feature Importance: Global impact of each feature on predictions

Correlation Heatmap: Relationships between different network features

Interactive Visualizations: Dynamic plots for data exploration

### Model Explanations
Model Comparison: Performance metrics across all models

SHAP Force Plots: Individual prediction explanations

Attention Weights: Neural network focus visualization

Prediction Analysis: Model consensus and confidence levels

### Threat Detection
Real-time Analysis: Live threat assessment interface

Manual Input Testing: Custom network feature testing

Multi-model Scoring: Threat scores from all models

Actionable Insights: Recommended security actions

### Technical Architecture
Data Processing Pipeline
text
Raw Network Data → Feature Engineering → Scaling/Normalization → Model Inference
Model Ensemble
XGBoost: Handles complex feature interactions

Neural Network: Captures deep patterns with attention

Ensemble Method: Combines strengths of all approaches

Explainability Framework
SHAP (SHapley Additive exPlanations): Model-agnostic interpretability

Attention Mechanisms: Built-in neural network interpretability

Comparative Analysis: Cross-model explanation validation

### Model Performance
Model	Accuracy	Precision	Recall	F1-Score
XGBoost Ensemble	98.7%	97.5%	96.8%	97.1%
Baseline XGBoost	97.5%	96.2%	95.4%	95.8%
Attention NN	96.8%	95.1%	94.7%	94.9%

 
 ### Usage Examples
Testing Normal Traffic
Use default feature values to simulate normal network behavior:

Low threat scores (< 0.3)

Green "LOW THREAT" indicators

Normal pattern explanations

Testing Attack Patterns
Simulate attacks with extreme values:

src_bytes: 10000+ (high data transfer)

num_failed_logins: 5+ (brute force attempts)

wrong_fragment: 2+ (fragmentation attacks)

High threat scores (> 0.7)

Red "CRITICAL THREAT" alerts

### Customization
Adding New Features
Update feature engineering in preprocessing notebooks

Retrain models with new feature set

Update feature_names.pkl

Dashboard automatically adapts to new features

Model Retraining
python
# Example retraining script
from notebooks.model_training import retrain_models
retrain_models(new_data_path='path/to/new_data.csv')

 ### Feature Description
The system analyzes 144 network traffic features including:

Basic Connection: protocol_type, service, flag, src_bytes, dst_bytes

Traffic Patterns: duration, src_count, dst_count, rate metrics

Security Indicators: logged_in, num_failed_logins, root_shell attempts

Statistical Features: same_srv_rate, diff_srv_rate, error rates

Time-based: connection frequency, service patterns, host behavior

### Threat Classification
Threat Level	Score Range	Description	Actions
LOW	0.0 - 0.4	Normal network activity	Continue monitoring
MEDIUM	0.4 - 0.7	Suspicious patterns	Investigate logs
HIGH	0.7 - 0.9	Likely malicious activity	Security review
CRITICAL	0.9 - 1.0	Confirmed attack	Immediate response

### Contributing
We welcome contributions! Please see our contributing guidelines for details.

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.


### Acknowledgments
XGBoost: Machine learning library

SHAP: Explainable AI framework

Streamlit: Web application framework

PyTorch: Deep learning library

Plotly: Interactive visualization library

### team members
Zainab Batool
Maryam Fatima
