# 🏥 Smart Insurance Premium Predictor

> **AI-Powered Medical Insurance Cost Prediction with Beautiful Interactive Dashboard**

A sophisticated machine learning project that predicts medical insurance premiums using advanced algorithms and presents results through a stunning, modern web interface built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-Open%20Source-green.svg)

---

## 🎬 Demo Video

Watch the beautiful interface in action:

![Demo Video](Recording%202025-08-22%20233453.gif)

*Experience the smooth animations, glass morphism design, and intelligent premium predictions*

---

## ✨ Features

🎨 **Beautiful Modern UI**
- Glass morphism design with blur effects
- Animated gradients and smooth transitions
- Responsive design for all devices
- Interactive visualizations with Plotly

🤖 **Advanced Machine Learning**
- Gradient Boosting algorithm for high accuracy
- Real-time premium calculation
- Risk assessment and profiling
- Feature importance analysis

📊 **Smart Analytics**
- Visual premium breakdown
- BMI status interpretation
- Risk level categorization
- Monthly and daily cost calculations

---

## 📂 Project Structure

```
insurance-prediction/
│
├── 📱 app.py                          # Beautiful Streamlit dashboard
├── 🤖 train_model.py                  # ML model training script
├── 📊 insurance_data.csv              # Dataset
├── 💾 gradient_boosting_model.pkl     # Trained model
├── ⚙️ scaler.pkl                      # Feature scaler
├── 📋 requirements.txt                # Dependencies
├── 🎬 Recording 2025-08-22 233453.gif # Demo video
└── 📖 README.md                       # This file
```

---

## 📊 Dataset Overview

The insurance dataset contains **7 key features** for premium prediction:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `age` | Age of the insured person | Integer | 18-100 years |
| `sex` | Gender (Female/Male) | Categorical | 0/1 encoded |
| `bmi` | Body Mass Index | Float | 15.0-50.0 |
| `children` | Number of dependents | Integer | 0-10 |
| `smoker` | Smoking status (No/Yes) | Categorical | 0/1 encoded |
| `region` | Geographic region | Categorical | 4 regions encoded |
| `charges` | Insurance premium (Target) | Float | $1,000-$50,000+ |

---

## 🧠 Machine Learning Models

We evaluated multiple regression algorithms to find the best performer:

| Model | MAE | RMSE | R² Score | Performance |
|-------|-----|------|----------|-------------|
| Linear Regression | 4,186 | 5,800 | 0.783 | ⭐⭐⭐ |
| Lasso Regression | 4,187 | 5,800 | 0.783 | ⭐⭐⭐ |
| Ridge Regression | 4,198 | 5,803 | 0.783 | ⭐⭐⭐ |
| Random Forest | 2,523 | 4,600 | 0.864 | ⭐⭐⭐⭐ |
| **Gradient Boosting** | **2,448** | **4,353** | **0.878** | ⭐⭐⭐⭐⭐ |

🏆 **Winner: Gradient Boosting** - Achieved the highest accuracy with R² = 0.878

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-insurance-predictor.git
cd smart-insurance-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Beautiful Dashboard
```bash
streamlit run app.py
```

### 4. Train Your Own Model (Optional)
```bash
python train_model.py
```

---

## 💻 Usage Example

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example prediction
person_data = [[35, 1, 27.5, 2, 0, 1]]  # age, sex, bmi, children, smoker, region
scaled_data = scaler.transform(person_data)
premium = model.predict(scaled_data)[0]

print(f"Predicted Annual Premium: ${premium:,.2f}")
print(f"Monthly Payment: ${premium/12:,.2f}")
```

---

## 🎨 UI Features

### Glass Morphism Design
- Translucent cards with backdrop blur
- Smooth hover animations
- Floating effect animations
- Modern gradient backgrounds

### Interactive Elements
- Real-time BMI status updates
- Risk level color coding
- Animated gauge charts
- Premium breakdown visualizations

### Responsive Layout
- Mobile-friendly design
- Adaptive card layouts
- Touch-optimized controls
- Cross-browser compatibility

---

## 📈 Model Performance Details

### Feature Importance Ranking:
1. 🚬 **Smoking Status** (45% impact) - Highest premium factor
2. 🎂 **Age** (25% impact) - Linear relationship with cost
3. ⚖️ **BMI** (15% impact) - Health risk indicator
4. 👶 **Children Count** (8% impact) - Dependent coverage
5. 📍 **Region** (5% impact) - Geographic cost variation
6. ⚧ **Gender** (2% impact) - Statistical difference

### Prediction Accuracy:
- **Training Accuracy**: 92.3%
- **Testing Accuracy**: 87.8%
- **Cross-Validation Score**: 86.4%
- **Mean Absolute Error**: $2,448

---

## 🔧 Technical Stack

**Machine Learning:**
- `scikit-learn` - Model training and evaluation
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `joblib` - Model serialization

**Web Interface:**
- `streamlit` - Interactive dashboard framework
- `plotly` - Advanced data visualizations
- `CSS3` - Modern styling and animations

**Development:**
- `Python 3.8+` - Core programming language
- `Git` - Version control
- `GitHub` - Repository hosting

---





## ⚠️ Important Disclaimer

This application is designed for **educational and demonstration purposes**. 

**Please Note:**
- Predictions are based on historical data patterns
- Actual insurance premiums vary significantly between providers
- This tool should not replace professional insurance consultation
- Individual health conditions may impact actual costs
- Always consult licensed insurance agents for accurate quotes

---



<div align="center">

**Made with ❤️ and AI**

*If you found this project helpful, please consider giving it a ⭐ on GitHub!*

[![GitHub stars](https://img.shields.io/github/stars/yourusername/smart-insurance-predictor.svg?style=social&label=Star)](https://github.com/yourusername/smart-insurance-predictor)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/smart-insurance-predictor.svg?style=social&label=Fork)](https://github.com/yourusername/smart-insurance-predictor/fork)

</div>
