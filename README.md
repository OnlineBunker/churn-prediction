# Customer Churn Prediction Dashboard

## Project Overview

This project presents an end-to-end Machine Learning system for predicting customer churn. The system analyzes historical behavioral and transactional data to identify customers at risk of leaving and provides actionable analytical insights.

The solution is implemented as an interactive Streamlit dashboard and is publicly deployed for demonstration.

---

## Objective

The primary objective of this project is to:

* Predict customer churn probability using supervised learning.
* Identify key drivers influencing churn behavior.
* Present insights through an interactive and visually structured dashboard.
* Deploy the solution as a publicly accessible web application.

---

## Technology Stack

* **Python**
* **Scikit-Learn** (Machine Learning Pipelines)
* **Pandas** (Data Processing)
* **Plotly** (Interactive Visualizations)
* **Streamlit** (Frontend Dashboard)

---

## Machine Learning Approach

### Data Preprocessing

We implemented a structured preprocessing pipeline using `ColumnTransformer`:

* StandardScaler for numerical features
* OneHotEncoder for categorical features

### Models Used

* Logistic Regression (Primary Model)
* Decision Tree (Comparative Evaluation)

### Evaluation Metrics

* ROC-AUC Score
* Confusion Matrix
* Churn Rate Analysis
* Feature Correlation Analysis

The model is encapsulated within a Scikit-Learn Pipeline to ensure reproducibility and seamless deployment.

---

## Dashboard Features

The deployed dashboard includes:

### 1. Overview

* Dataset preview
* ROC-AUC performance metric
* Confusion matrix visualization
* Key statistics summary

### 2. Exploratory Data Analysis (EDA)

* Churn rate by categorical features
* Numerical feature distribution by churn
* Behavioral trend insights

### 3. Prediction System

* Customer-level churn probability predictions
* High-risk customer identification
* Correlation with churn visualization

The interface is designed with a modern dark matte theme and neon-accent visualization style for improved clarity and presentation quality.

---

## Deployment

The application is deployed using Streamlit Cloud and integrated directly with this GitHub repository.

Public Hosted Link:
https://churn-prediction-web.streamlit.app

---

## Repository Structure

```
customer-churn-dashboard/
│
├── app.py
├── pipeline.pkl
├── requirements.txt
├── README.md
```

---

## Installation (Local Setup)

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Academic Context

This project was developed as part of an academic milestone focusing on classical machine learning pipelines for customer churn prediction. The system emphasizes structured preprocessing, modular design, evaluation rigor, and user-facing deployment.

---

## Future Scope

Potential enhancements include:

* Agentic retention strategy integration
* RAG-based retention recommendation system
* Automated intervention planning
* Expanded model comparison and explainability techniques

---

## License

This project is developed for academic purposes.
