import os
import sys

# Ensure the repo root is always on sys.path so 'agent' and 'rag' are importable
# regardless of the working directory Streamlit uses at runtime.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# ---------------------------------------------------
# API Configuration
# Priority: Streamlit secrets (cloud) → .env (local)
# ---------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required on Streamlit Cloud

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Customer Churn AI System", layout="wide")

# ---------------------------------------------------
# Custom Matte Neon Theme
# ---------------------------------------------------
NEON_COLORS = [
    "#00F5D4",  # neon teal
    "#F15BB5",  # neon pink
    "#9B5DE5",  # neon purple
    "#00BBF9",  # neon blue
    "#FEE440",  # neon yellow
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
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to   {opacity: 1; transform: translateY(0);}
}
.risk-critical { background-color: #3D0000; border-left: 4px solid #FF3333; padding: 12px; border-radius: 6px; }
.risk-high     { background-color: #3D1A00; border-left: 4px solid #FF8C00; padding: 12px; border-radius: 6px; }
.risk-medium   { background-color: #2E2E00; border-left: 4px solid #FFD700; padding: 12px; border-radius: 6px; }
.risk-low      { background-color: #003D10; border-left: 4px solid #00C853; padding: 12px; border-radius: 6px; }
.agent-step    { background-color: #1A1D24; border-radius: 8px; padding: 14px; margin-bottom: 10px; }
.report-box    { background-color: #12151C; border: 1px solid #00F5D4; border-radius: 10px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Customer Churn AI System")
st.caption("Milestone 1: ML Prediction  |  Milestone 2: Agentic Retention Strategist")

# ---------------------------------------------------
# Load Model & Data
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("pipeline.pkl")

model = load_model()

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
# Tabs  (4 tabs total)
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 EDA",
    "🎯 Prediction System",
    "🤖 Agentic Retention AI",
])

# ===================================================
# TAB 1 - OVERVIEW (Milestone 1)
# ===================================================
with tab1:
    st.subheader("Project Overview")
    st.write("""
    This system predicts customer churn using Logistic Regression
    with preprocessing (StandardScaler + OneHotEncoder) inside a Scikit-Learn Pipeline.
    Milestone 2 adds a LangGraph-based agentic retention strategy assistant.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df_eda.head())
    st.markdown("---")

    progress = st.progress(0)
    for i in range(100):
        progress.progress(i + 1)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = model.predict(X)
    roc     = roc_auc_score(y, y_proba)

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC Score", round(roc, 4))
    col2.metric("Total Customers", len(df))

    st.markdown("---")
    cm = confusion_matrix(y, y_pred)
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Viridis",
        title="Confusion Matrix", template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ===================================================
# TAB 2 - EDA (Milestone 1)
# ===================================================
with tab2:
    st.subheader("Churn Rate by Categorical Features")
    categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
    for col in categorical_cols:
        churn_rate = (
            df_eda.groupby(col)["Churn"]
            .mean().reset_index()
            .sort_values("Churn", ascending=False)
        )
        fig = px.bar(
            churn_rate, x=col, y="Churn",
            title=f"Churn Rate by {col}",
            text=churn_rate["Churn"].round(3),
            template=PLOT_TEMPLATE, color=col,
            color_discrete_sequence=NEON_COLORS,
        )
        fig.update_layout(yaxis_title="Churn Rate", xaxis_title=col)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Numerical Feature Distribution by Churn")
    numerical_cols = [
        "Age", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay",
        "Total Spend", "Last Interaction",
    ]
    df_eda["Churn_Label"] = df_eda["Churn"].map({0: "No", 1: "Yes"})
    for col in numerical_cols:
        fig = px.box(
            df_eda, x="Churn_Label", y=col, color="Churn_Label",
            title=f"{col} Distribution by Churn",
            template=PLOT_TEMPLATE,
            color_discrete_sequence=["#F15BB5", "#00F5D4"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ===================================================
# TAB 3 - PREDICTION SYSTEM (Milestone 1)
# ===================================================
with tab3:
    st.subheader("Prediction Results")

    X        = df.drop(columns=["Churn"])
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df_results = df.copy()
    df_results["Churn Prediction"]  = predictions
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
        "Total Spend", "Last Interaction",
    ]
    target_corr = (
        df_eda[numerical_cols + ["Churn"]]
        .corr()["Churn"].drop("Churn")
        .sort_values(ascending=False)
    )
    target_corr_df = target_corr.reset_index()
    target_corr_df.columns = ["Feature", "Correlation"]

    fig_corr = px.bar(
        target_corr_df, x="Correlation", y="Feature", orientation="h",
        title="Correlation with Churn", template=PLOT_TEMPLATE,
        color="Correlation", color_continuous_scale="Tealgrn",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.sidebar:
        st.markdown("## 📊 Quick Stats")
        st.metric("Total Customers", len(df))
        st.metric("Churn Rate", f"{round(df['Churn'].mean() * 100, 2)}%")
        st.metric("High Risk Customers", len(high_risk))

# ===================================================
# TAB 4 - AGENTIC RETENTION AI  (Milestone 2)
# ===================================================
with tab4:

    # ── Hero Banner ───────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0D1B2A 0%, #1A1D2E 50%, #0D1B2A 100%);
        border: 1px solid #00F5D4;
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -40px; right: -40px;
            width: 200px; height: 200px;
            background: radial-gradient(circle, rgba(0,245,212,0.08) 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute; bottom: -30px; left: 20%;
            width: 150px; height: 150px;
            background: radial-gradient(circle, rgba(155,93,229,0.07) 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:12px;">
            <span style="font-size:2.4rem;">🤖</span>
            <h1 style="color:#00F5D4; margin:0; font-size:1.9rem; letter-spacing:0.5px;">
                Agentic Retention Strategy AI
            </h1>
        </div>
        <p style="color:#B0BEC5; font-size:1.05rem; margin:0; line-height:1.7;">
            Powered by <strong style="color:#00F5D4;">LangGraph</strong> &nbsp;·&nbsp;
            <strong style="color:#9B5DE5;">RAG Retrieval</strong> &nbsp;·&nbsp;
            <strong style="color:#F15BB5;">Groq LLM (llama3-8b)</strong>
            <br>Autonomously analyzes churn risk, retrieves best-fit strategies, and generates
            a structured retention action plan — in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Visual ───────────────────────────────
    with st.expander("📐 How the Agent Works", expanded=False):
        st.markdown("""
        <div style="display:flex; justify-content:center; align-items:center; gap:0; flex-wrap:wrap; padding:12px 0;">
            <div style="background:#1A1D24; border:1px solid #00F5D4; border-radius:10px; padding:14px 20px; text-align:center; min-width:130px;">
                <div style="font-size:1.4rem;">👤</div>
                <div style="color:#00F5D4; font-weight:600; font-size:0.85rem;">Customer Input</div>
                <div style="color:#777; font-size:0.75rem;">Profile + Churn Prob</div>
            </div>
            <div style="color:#00F5D4; font-size:1.4rem; padding:0 6px;">→</div>
            <div style="background:#1A1D24; border:1px solid #FF8C00; border-radius:10px; padding:14px 20px; text-align:center; min-width:130px;">
                <div style="font-size:1.4rem;">🔍</div>
                <div style="color:#FF8C00; font-weight:600; font-size:0.85rem;">Node 1</div>
                <div style="color:#777; font-size:0.75rem;">Risk Analyzer</div>
            </div>
            <div style="color:#00F5D4; font-size:1.4rem; padding:0 6px;">→</div>
            <div style="background:#1A1D24; border:1px solid #9B5DE5; border-radius:10px; padding:14px 20px; text-align:center; min-width:130px;">
                <div style="font-size:1.4rem;">📚</div>
                <div style="color:#9B5DE5; font-weight:600; font-size:0.85rem;">Node 2</div>
                <div style="color:#777; font-size:0.75rem;">RAG Retriever</div>
            </div>
            <div style="color:#00F5D4; font-size:1.4rem; padding:0 6px;">→</div>
            <div style="background:#1A1D24; border:1px solid #F15BB5; border-radius:10px; padding:14px 20px; text-align:center; min-width:130px;">
                <div style="font-size:1.4rem;">✍️</div>
                <div style="color:#F15BB5; font-weight:600; font-size:0.85rem;">Node 3</div>
                <div style="color:#777; font-size:0.75rem;">Groq LLM Report</div>
            </div>
            <div style="color:#00F5D4; font-size:1.4rem; padding:0 6px;">→</div>
            <div style="background:#1A1D24; border:1px solid #00F5D4; border-radius:10px; padding:14px 20px; text-align:center; min-width:130px;">
                <div style="font-size:1.4rem;">📄</div>
                <div style="color:#00F5D4; font-weight:600; font-size:0.85rem;">Output</div>
                <div style="color:#777; font-size:0.75rem;">Retention Report</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # LEFT PANEL: Customer Setup  |  RIGHT: Query + Run
    # ═══════════════════════════════════════════════
    setup_col, query_col = st.columns([3, 2], gap="large")

    customer_profile = {}
    churn_prob = 0.5

    with setup_col:
        st.markdown("""
        <div style="background:#13161F; border:1px solid #2A2D3A; border-radius:12px; padding:20px 24px; margin-bottom:4px;">
            <h3 style="color:#00F5D4; margin-top:0; margin-bottom:16px; font-size:1.1rem; letter-spacing:0.3px;">
                🎛️ Customer Profile Input
            </h3>
        """, unsafe_allow_html=True)

        input_mode = st.radio(
            "Input Mode",
            ["📋 Manual Input", "📂 Select from Dataset"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if input_mode == "📋 Manual Input":
            st.markdown("<div style='background:#13161F; border:1px solid #2A2D3A; border-radius:12px; padding:20px 24px; margin-top:12px;'>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                age           = st.slider("Age", 18, 80, 35)
                tenure        = st.slider("Tenure (months)", 1, 60, 12)
                usage_freq    = st.slider("Usage Frequency", 1, 30, 10)
                support_calls = st.slider("Support Calls", 0, 15, 3)
            with c2:
                payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
                total_spend   = st.number_input("Total Spend ($)", 50, 2000, 400)
                last_interact = st.slider("Last Interaction (days ago)", 1, 30, 10)

            r1, r2, r3 = st.columns(3)
            gender   = r1.selectbox("Gender",            ["Male", "Female"])
            sub_type = r2.selectbox("Subscription",      ["Basic", "Standard", "Premium"])
            contract = r3.selectbox("Contract",          ["Monthly", "Quarterly", "Annual"])

            st.markdown("</div>", unsafe_allow_html=True)

            customer_profile = {
                "Age": age, "Tenure": tenure, "Usage Frequency": usage_freq,
                "Support Calls": support_calls, "Payment Delay": payment_delay,
                "Total Spend": total_spend, "Last Interaction": last_interact,
                "Gender": gender, "Subscription Type": sub_type, "Contract Length": contract,
            }
            try:
                input_df   = pd.DataFrame([customer_profile])
                churn_prob = float(model.predict_proba(input_df)[0][1])
            except Exception:
                churn_prob = 0.5

        else:
            st.markdown("<div style='background:#13161F; border:1px solid #2A2D3A; border-radius:12px; padding:20px 24px; margin-top:12px;'>", unsafe_allow_html=True)

            df_display = df_eda.copy()
            X_all  = df.drop(columns=["Churn"])
            p_all  = model.predict_proba(X_all)[:, 1]
            df_display["Churn Probability"] = p_all
            df_display = df_display.sort_values("Churn Probability", ascending=False)

            customer_idx = st.selectbox(
                "Select Customer (sorted by churn risk)",
                options=df_display.index.tolist(),
                format_func=lambda i: (
                    f"#{i}  |  Risk: {df_display.loc[i,'Churn Probability']:.1%}  "
                    f"|  Tenure: {df_display.loc[i,'Tenure']}mo  "
                    f"|  {df_display.loc[i,'Subscription Type']}"
                ),
            )
            row        = df_display.loc[customer_idx]
            churn_prob = float(row["Churn Probability"])
            customer_profile = {
                col: row[col]
                for col in [
                    "Age", "Tenure", "Usage Frequency", "Support Calls",
                    "Payment Delay", "Total Spend", "Last Interaction",
                    "Gender", "Subscription Type", "Contract Length",
                ]
                if col in row.index
            }

            profile_df = pd.DataFrame(list(customer_profile.items()), columns=["Feature", "Value"])
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with query_col:
        # ── Live Risk Gauge ───────────────────────────
        prob_pct = churn_prob * 100
        if churn_prob >= 0.80:
            tier_label = "CRITICAL"; tier_color = "#FF3333"; tier_bg = "#3D0000"; tier_border = "#FF3333"
        elif churn_prob >= 0.60:
            tier_label = "HIGH";     tier_color = "#FF8C00"; tier_bg = "#3D1A00"; tier_border = "#FF8C00"
        elif churn_prob >= 0.35:
            tier_label = "MEDIUM";   tier_color = "#FFD700"; tier_bg = "#2E2E00"; tier_border = "#FFD700"
        else:
            tier_label = "LOW";      tier_color = "#00C853"; tier_bg = "#003D10"; tier_border = "#00C853"

        bar_width = int(prob_pct)

        st.markdown(f"""
        <div style="
            background:{tier_bg};
            border:1px solid {tier_border};
            border-radius:14px;
            padding:24px;
            text-align:center;
            margin-bottom:16px;
        ">
            <p style="color:#AAAAAA; margin:0 0 4px 0; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase;">Churn Risk Level</p>
            <h1 style="color:{tier_color}; font-size:2.6rem; margin:0; line-height:1.1;">{prob_pct:.1f}%</h1>
            <span style="
                display:inline-block;
                background:{tier_color}22;
                color:{tier_color};
                border:1px solid {tier_color};
                border-radius:20px;
                padding:3px 18px;
                font-size:0.85rem;
                font-weight:700;
                letter-spacing:2px;
                margin-top:8px;
            ">{tier_label}</span>
            <div style="background:#1A1A1A; border-radius:6px; height:8px; margin-top:16px; overflow:hidden;">
                <div style="
                    background: linear-gradient(90deg, {tier_color}88, {tier_color});
                    width:{bar_width}%;
                    height:100%;
                    border-radius:6px;
                    transition: width 0.5s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Chat / Query Box ──────────────────────────
        st.markdown("""
        <div style="
            background: linear-gradient(145deg, #0E1420, #141828);
            border: 1px solid #00F5D440;
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 14px;
        ">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span style="font-size:1.3rem;">💬</span>
                <span style="color:#00F5D4; font-weight:600; font-size:1rem;">Ask the AI Agent</span>
            </div>
            <p style="color:#778899; font-size:0.83rem; margin:0 0 10px 0; line-height:1.5;">
                Optionally ask a specific retention question or leave blank for a full analysis report.
            </p>
        </div>
        """, unsafe_allow_html=True)

        user_query = st.text_area(
            "Your question",
            placeholder="e.g. What's the best discount strategy for a Basic plan customer with high support calls?\n\nOr leave blank for a full retention strategy report...",
            height=110,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Run Button ────────────────────────────────
        run_btn = st.button(
            "🚀  Analyze & Generate Retention Report",
            type="primary",
            use_container_width=True,
        )

    # ═══════════════════════════════════════════════
    # AGENT EXECUTION & RESULTS
    # ═══════════════════════════════════════════════
    if run_btn:
        try:
            from retention_agent import run_retention_agent
        except ImportError as e:
            st.error(
                f"Agent import failed: {e}\n\n"
                "Make sure `retention_agent.py` and other required files are present."
            )
            st.stop()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="border-top: 1px solid #2A2D3A; margin: 10px 0 24px 0;"></div>
        <h3 style="color:#00F5D4; letter-spacing:0.5px;">⚡ Agent Execution</h3>
        """, unsafe_allow_html=True)

        with st.status("🤖 LangGraph agent running...", expanded=True) as status:
            st.write("🔍 **Step 1 / 3** — Risk Analyzer node: classifying churn tier and signals...")
            st.write("📚 **Step 2 / 3** — RAG Retriever node: fetching relevant retention strategies...")
            st.write("✍️ **Step 3 / 3** — Groq LLM node: synthesizing structured retention report...")
            result = run_retention_agent(
                customer_profile=customer_profile,
                churn_probability=churn_prob,
                query=user_query,
            )
            status.update(label="✅ Agent completed successfully!", state="complete")

        if result.get("error"):
            st.warning(f"⚠️ {result['error']}")

        # ── Three-column result cards ─────────────────
        res1, res2, res3 = st.columns(3)
        docs = result.get("retrieved_docs", [])
        with res1:
            st.markdown(f"""
            <div style="background:#13161F; border:1px solid #FF8C00; border-radius:10px; padding:16px; text-align:center;">
                <p style="color:#888; font-size:0.75rem; margin:0 0 4px 0; text-transform:uppercase; letter-spacing:1px;">Risk Tier</p>
                <h2 style="color:#FF8C00; margin:0; font-size:1.8rem;">{result.get('risk_tier','—')}</h2>
            </div>""", unsafe_allow_html=True)
        with res2:
            st.markdown(f"""
            <div style="background:#13161F; border:1px solid #9B5DE5; border-radius:10px; padding:16px; text-align:center;">
                <p style="color:#888; font-size:0.75rem; margin:0 0 4px 0; text-transform:uppercase; letter-spacing:1px;">Strategies Retrieved</p>
                <h2 style="color:#9B5DE5; margin:0; font-size:1.8rem;">{len(docs)}</h2>
            </div>""", unsafe_allow_html=True)
        with res3:
            st.markdown(f"""
            <div style="background:#13161F; border:1px solid #00F5D4; border-radius:10px; padding:16px; text-align:center;">
                <p style="color:#888; font-size:0.75rem; margin:0 0 4px 0; text-transform:uppercase; letter-spacing:1px;">Report Length</p>
                <h2 style="color:#00F5D4; margin:0; font-size:1.8rem;">{len(result.get('retention_report',''))} ch</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Summary ──────────────────────────────
        st.markdown("""<h4 style="color:#EAEAEA; margin-bottom:10px;">🔎 Risk Assessment Summary</h4>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#1A1D24; border-left:4px solid {tier_border}; border-radius:8px; padding:16px 20px; margin-bottom:20px; line-height:1.7; color:#CCCCCC;">
            {result['risk_summary']}
        </div>""", unsafe_allow_html=True)

        # ── RAG Results ───────────────────────────────
        st.markdown("""<h4 style="color:#EAEAEA; margin-bottom:10px;">📚 Retrieved Retention Strategies (RAG)</h4>""", unsafe_allow_html=True)
        if docs:
            rag_cols = st.columns(len(docs))
            rag_colors = ["#9B5DE5", "#00BBF9", "#F15BB5"]
            for i, (col, doc) in enumerate(zip(rag_cols, docs)):
                with col:
                    st.markdown(f"""
                    <div style="
                        background:#13161F;
                        border:1px solid {rag_colors[i]};
                        border-radius:10px;
                        padding:14px;
                        height:100%;
                    ">
                        <p style="color:{rag_colors[i]}; font-weight:700; font-size:0.82rem; margin:0 0 8px 0; text-transform:uppercase; letter-spacing:0.5px;">
                            Strategy {i+1} &nbsp;·&nbsp; {doc['score']:.2f}
                        </p>
                        <p style="color:#EAEAEA; font-size:0.88rem; font-weight:600; margin:0 0 8px 0; line-height:1.4;">{doc['title']}</p>
                        <p style="color:#9AA5B1; font-size:0.78rem; margin:0; line-height:1.5;">{doc['content'][:160]}...</p>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No documents retrieved.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Main Report ───────────────────────────────
        st.markdown("""<h4 style="color:#EAEAEA; margin-bottom:12px;">📄 AI-Generated Retention Strategy Report</h4>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="
            background: linear-gradient(160deg, #0D1220 0%, #111827 100%);
            border: 1px solid #00F5D4;
            border-radius: 14px;
            padding: 28px 32px;
            line-height: 1.85;
            color: #D1D5DB;
            font-size: 0.95rem;
        ">
            {result['retention_report'].replace(chr(10), '<br>')}
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Actions Row ───────────────────────────────
        dl_col, trace_col = st.columns([1, 1])
        with dl_col:
            st.download_button(
                label="⬇️  Download Report (.md)",
                data=result["retention_report"],
                file_name=f"retention_report_{tier_label.lower()}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with trace_col:
            with st.expander("🗺️ Agent Workflow Trace"):
                st.markdown(f"""
                | Step | Node | Status |
                |------|------|--------|
                | 1 | `analyze_risk_node` | ✅ Risk Tier: **{result['risk_tier']}** |
                | 2 | `retrieve_strategies_node` | ✅ Retrieved **{len(docs)}** docs |
                | 3 | `generate_report_node` | ✅ {len(result.get('retention_report',''))} chars generated |
                """)

    # ── Ethical Disclaimer ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background:#0D1117;
        border:1px solid #2A2D3A;
        border-radius:10px;
        padding:14px 20px;
        color:#666;
        font-size:0.8rem;
        line-height:1.6;
    ">
        ⚠️ <strong style="color:#888;">Ethical Disclaimer:</strong>
        This AI system provides probabilistic predictions for business decision support only.
        All recommendations must be reviewed by a human customer success manager before execution.
        Customer data is processed in compliance with applicable data privacy regulations (GDPR/CCPA).
        The system does not make autonomous decisions affecting customers without human oversight.
    </div>
    """, unsafe_allow_html=True)
