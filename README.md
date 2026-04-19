# Customer Churn Prediction & Agentic Retention Strategy

## Project Overview

An end-to-end AI system that predicts customer churn (Milestone 1) and evolves into an
agentic AI retention strategist (Milestone 2).

---

## 🏗️ System Architecture

### Milestone 1: ML-Based Churn Prediction
- **Models**: Logistic Regression + Decision Tree (Scikit-Learn Pipeline)
- **Preprocessing**: StandardScaler + OneHotEncoder via ColumnTransformer
- **Evaluation**: ROC-AUC, Confusion Matrix, Correlation Analysis
- **UI**: Streamlit dashboard with EDA and batch prediction

### Milestone 2: Agentic Retention AI
- **Framework**: LangGraph (directed graph with explicit state management)
- **LLM**: Groq API → `llama3-8b-8192` (free tier)
- **RAG**: Custom FAISS-style TF-IDF vector retriever over retention knowledge base
- **Agent Nodes**:
  1. `analyze_risk_node` — Deterministic risk tier classification (CRITICAL/HIGH/MEDIUM/LOW)
  2. `retrieve_strategies_node` — Semantic retrieval of top-3 retention strategies
  3. `generate_report_node` — Groq LLM report synthesis with anti-hallucination prompting

---

## 📁 Repository Structure

```
churn-prediction/
│
├── app.py                          # Streamlit app (Milestone 1 + 2)
├── pipeline.pkl                    # Trained ML pipeline
├── data.csv                        # Customer dataset
├── requirements.txt
├── README.md
│
├── agent/
│   ├── __init__.py
│   └── retention_agent.py          # LangGraph agent (3-node workflow)
│
└── rag/
    ├── __init__.py
    ├── retention_knowledge.py      # Curated retention strategy knowledge base
    └── retriever.py                # TF-IDF FAISS-style vector retriever
```

---

## 🚀 Deployment

**Live App**: [churn-prediction-web.streamlit.app](https://churn-prediction-web.streamlit.app)

### Local Setup

```bash
git clone https://github.com/OnlineBunker/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables

Create a `.streamlit/secrets.toml` for Streamlit Cloud deployment:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Or set as an environment variable locally:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

---

## 🤖 Milestone 2: How the Agent Works

1. **Input**: Customer profile + churn probability from ML model
2. **Node 1 — Risk Analyzer**: Classifies risk tier, identifies behavioral signals
3. **Node 2 — RAG Retriever**: Builds semantic query, retrieves top-3 retention docs
4. **Node 3 — Report Generator**: Groq LLM synthesizes structured Markdown report
5. **Output**: Risk summary, retrieved strategies, structured retention report

### Anti-Hallucination Measures
- LLM is instructed to use ONLY retrieved documents as its source
- Temperature set to 0.3 (low randomness)
- Graceful fallback to rule-based report if LLM fails
- Ethical disclaimer on all outputs

---

## 📊 Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Models | Scikit-Learn (LR + Decision Tree) |
| Agent Framework | LangGraph |
| LLM | Groq API (llama3-8b-8192) |
| RAG | Custom TF-IDF retriever |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |

---

## ⚠️ Ethical Disclaimer

This AI system provides probabilistic predictions for business decision support only.
All recommendations must be reviewed by a human customer success manager before execution.
Customer data is processed in compliance with applicable data privacy regulations.

---

## Academic Context

Developed for the GenAI & Agentic AI course — Project 5: Customer Churn Prediction & Agentic Retention Strategy.
