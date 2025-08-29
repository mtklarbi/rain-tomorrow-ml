# 🌦️ Rain Tomorrow in Tiznit — End-to-End ML Project

**Goal:** Predict whether it will rain tomorrow in Tiznit using historical weather data.

## 🔹 Pipeline
1. **Data collection**: Downloaded daily weather data via the [Open-Meteo API](https://open-meteo.com/).
2. **Preprocessing**: Cleaning, target creation (`rain_tomorrow`), rolling averages (3 & 7 days), time-based train/test split.
3. **Modeling**: Trained Logistic Regression (with scaling) and RandomForest classifiers (scikit-learn).
4. **Evaluation**: Accuracy, Precision, Recall, and ROC-AUC (with ROC plots).
5. **Deployment**: Saved best model (joblib) and exposed it through a FastAPI endpoint (`/predict`).

## 📂 Structure

src/
├─ collect.py # Fetch weather data from API
├─ preprocess.py # Clean + feature engineering + split
├─ train.py # Train multiple models
├─ eval.py # Evaluate models with metrics + ROC curve
└─ app.py # Minimal FastAPI app for deployment
data/ # CSVs + train/test sets
models/ # Saved joblib models


## 🛠️ Tech Stack
- **Python**: pandas, numpy, scikit-learn, matplotlib, joblib, requests  
- **API**: FastAPI + Uvicorn  
- **Workflow**: modular Python scripts, end-to-end reproducible pipeline  

---

✅ This project shows ability to **design, train, evaluate and deploy** a complete ML pipeline.
