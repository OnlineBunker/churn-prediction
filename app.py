import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------------------------------------------------
# Custom Matte Neon Theme
# ---------------------------------------------------
NEON_COLORS = [
    "#00F5D4",  # neon teal
    "#F15BB5",  # neon pink
    "#9B5DE5",  # neon purple
    "#00BBF9",  # neon blue
    "#FEE440"   # neon yellow
]

PLOT_TEMPLATE = "plotly_dark"

st.markdown("""
<style>
    .stApp {
        background-color: #0F1117;
        color: #EAEAEA;
    }
    h1, h2, h3 {
        color: #00F5D4;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        padding: 10px;
    }
    .stMetric {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric {
        animation: fadeIn 0.6s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction System")

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
model = joblib.load("pipeline.pkl")

# ---------------------------------------------------
# Load Dataset (Cached)
# ---------------------------------------------------
url = "https://raw.githubusercontent.com/OnlineBunker/churn-prediction/main/data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()
df_eda = df.copy()

if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Prediction System"])

# ===================================================
# TAB 1 - OVERVIEW
# ===================================================
with tab1:

    st.subheader("Project Overview")

    st.write("""
    This system predicts customer churn using Logistic Regression 
    with preprocessing (StandardScaler + OneHotEncoder) inside a Scikit-Learn Pipeline.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df_eda.head())

    st.markdown("---")

    # Animated Progress Bar
    progress = st.progress(0)
    for i in range(100):
        progress.progress(i + 1)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    roc = roc_auc_score(y, y_proba)

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC Score", round(roc, 4))
    col2.metric("Total Customers", len(df))

    st.markdown("---")

    cm = confusion_matrix(y, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Confusion Matrix",
        template=PLOT_TEMPLATE
    )

    st.plotly_chart(fig_cm, use_container_width=True)


# ===================================================
# TAB 2 - EDA
# ===================================================
with tab2:

    st.subheader("Churn Rate by Categorical Features")

    categorical_cols = ["Gender", "Subscription Type", "Contract Length"]

    for col in categorical_cols:

        churn_rate = (
            df_eda.groupby(col)["Churn"]
            .mean()
            .reset_index()
            .sort_values("Churn", ascending=False)
        )

        fig = px.bar(
            churn_rate,
            x=col,
            y="Churn",
            title=f"Churn Rate by {col}",
            text=churn_rate["Churn"].round(3),
            template=PLOT_TEMPLATE,
            color=col,
            color_discrete_sequence=NEON_COLORS
        )

        fig.update_layout(
            yaxis_title="Churn Rate",
            xaxis_title=col
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Numerical Feature Distribution by Churn")

    numerical_cols = [
        "Age", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay",
        "Total Spend", "Last Interaction"
    ]

    df_eda["Churn_Label"] = df_eda["Churn"].map({0: "No", 1: "Yes"})

    for col in numerical_cols:

        fig = px.box(
            df_eda,
            x="Churn_Label",
            y=col,
            color="Churn_Label",
            title=f"{col} Distribution by Churn",
            template=PLOT_TEMPLATE,
            color_discrete_sequence=["#F15BB5", "#00F5D4"]
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)


# ===================================================
# TAB 3 - PREDICTION SYSTEM
# ===================================================
with tab3:

    st.subheader("Prediction Results")

    X = df.drop(columns=["Churn"])

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df_results = df.copy()
    df_results["Churn Prediction"] = predictions
    df_results["Churn Probability"] = probabilities

    st.dataframe(df_results)

    st.markdown("---")

    st.subheader("High Risk Customers (Probability > 0.7)")
    high_risk = df_results[df_results["Churn Probability"] > 0.7]
    st.dataframe(high_risk)

    st.markdown("---")

    st.subheader("Correlation with Churn")

    numerical_cols = [
        "Age", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay",
        "Total Spend", "Last Interaction"
    ]

    target_corr = (
        df_eda[numerical_cols + ["Churn"]]
        .corr()["Churn"]
        .drop("Churn")
        .sort_values(ascending=False)
    )

    target_corr_df = target_corr.reset_index()
    target_corr_df.columns = ["Feature", "Correlation"]

    fig_corr = px.bar(
        target_corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Correlation with Churn",
        template=PLOT_TEMPLATE,
        color="Correlation",
        color_continuous_scale="Tealgrn"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

with st.sidebar:
    st.markdown("## 📊 Quick Stats")
    st.metric("Total Customers", len(df))
    st.metric("Churn Rate", round(df["Churn"].mean() * 100, 2))
    st.metric("High Risk Customers", len(high_risk))