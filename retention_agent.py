"""
Agentic Retention Strategy Assistant
Milestone 2 - LangGraph Workflow

Graph:
  [START]
     |
     v
  analyze_risk_node        <- Reads customer profile, classifies risk tier
     |
     v
  retrieve_strategies_node  <- RAG: fetches relevant retention docs
     |
     v
  generate_report_node      <- Groq LLM: synthesizes structured report
     |
     v
  [END]

State is explicitly typed and passed through each node.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from retriever import get_rag


# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    customer_profile: dict[str, Any]
    churn_probability: float
    query: str                          # optional user question/context

    # Intermediate
    risk_tier: str                      # LOW / MEDIUM / HIGH / CRITICAL
    risk_summary: str                   # 2–3 sentence human-readable summary
    retrieved_docs: list[dict]          # RAG results

    # Output
    retention_report: str               # Final structured Markdown report
    error: str                          # Error message if any step fails


# ─────────────────────────────────────────────
# Node 1: Risk Analyzer
# ─────────────────────────────────────────────

def analyze_risk_node(state: AgentState) -> AgentState:
    """
    Classifies the customer into a churn risk tier and produces
    a structured risk summary for downstream nodes.
    Does NOT call LLM — pure deterministic logic to save API budget.
    """
    prob = state["churn_probability"]
    profile = state["customer_profile"]

    # Risk tier classification
    if prob >= 0.80:
        tier = "CRITICAL"
    elif prob >= 0.60:
        tier = "HIGH"
    elif prob >= 0.35:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Build risk signals
    signals = []
    if profile.get("Support Calls", 0) >= 5:
        signals.append("high support call volume")
    if profile.get("Payment Delay", 0) >= 20:
        signals.append("significant payment delays")
    if profile.get("Usage Frequency", 0) <= 5:
        signals.append("very low product usage")
    if profile.get("Tenure", 0) <= 6:
        signals.append("short customer tenure (new customer)")
    if profile.get("Last Interaction", 0) >= 25:
        signals.append("low recent interaction")
    if profile.get("Total Spend", 0) <= 200:
        signals.append("low total spend")

    signal_text = ", ".join(signals) if signals else "no critical behavioral signals detected"

    summary = (
        f"Customer exhibits a **{tier}** churn risk with a predicted probability of "
        f"**{prob:.1%}**. "
        f"Key risk signals include: {signal_text}. "
        f"Tenure: {profile.get('Tenure', 'N/A')} months | "
        f"Subscription: {profile.get('Subscription Type', 'N/A')} | "
        f"Contract: {profile.get('Contract Length', 'N/A')}."
    )

    return {
        **state,
        "risk_tier": tier,
        "risk_summary": summary,
    }


# ─────────────────────────────────────────────
# Node 2: RAG Retriever
# ─────────────────────────────────────────────

def retrieve_strategies_node(state: AgentState) -> AgentState:
    """
    Builds a semantic query from the customer profile + risk tier,
    then retrieves top-3 retention strategy documents from the RAG index.
    """
    try:
        profile = state["customer_profile"]
        tier = state["risk_tier"]
        prob = state["churn_probability"]

        # Construct a rich query for retrieval
        query_parts = [f"{tier} churn risk customer"]

        if profile.get("Support Calls", 0) >= 5:
            query_parts.append("high support calls service quality")
        if profile.get("Payment Delay", 0) >= 20:
            query_parts.append("payment delay billing")
        if profile.get("Usage Frequency", 0) <= 5:
            query_parts.append("low usage re-engagement product adoption")
        if profile.get("Tenure", 0) <= 6:
            query_parts.append("new customer onboarding early tenure")
        if profile.get("Tenure", 0) >= 48:
            query_parts.append("long tenure contract renewal loyalty")
        if prob >= 0.80:
            query_parts.append("immediate intervention win back critical")

        sub_type = profile.get("Subscription Type", "")
        if sub_type:
            query_parts.append(f"{sub_type} subscription plan upgrade")

        query = " ".join(query_parts)
        user_query = state.get("query", "")
        if user_query:
            query = user_query + " " + query

        rag = get_rag()
        docs = rag.retrieve(query, top_k=3)

        return {**state, "retrieved_docs": docs}

    except Exception as e:
        return {**state, "retrieved_docs": [], "error": f"RAG retrieval error: {str(e)}"}


# ─────────────────────────────────────────────
# Node 3: Report Generator (Groq LLM)
# ─────────────────────────────────────────────

def generate_report_node(state: AgentState) -> AgentState:
    """
    Uses Groq LLM to synthesize a structured retention report
    from the risk analysis and retrieved strategies.
    Includes anti-hallucination prompting (grounded in retrieved docs only).
    """
    try:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            return {
                **state,
                "retention_report": _fallback_report(state),
                "error": "GROQ_API_KEY not set — showing rule-based report.",
            }

        client = Groq(api_key=groq_key)

        profile = state["customer_profile"]
        docs = state["retrieved_docs"]

        # Format retrieved docs as grounding context
        context_text = "\n\n".join(
            f"[Strategy {i+1}: {d['title']}]\n{d['content']}"
            for i, d in enumerate(docs)
        ) if docs else "No specific strategies retrieved."

        system_prompt = """You are an expert Customer Retention Strategist AI.
Your task is to generate a structured, professional retention report.

STRICT RULES:
1. Base ALL recommendations ONLY on the provided retention strategies context below.
2. Do NOT invent statistics, policies, or strategies not present in the context.
3. If a recommendation is not supported by the context, write "Consult your customer success team."
4. Always include an ethical disclaimer at the end.
5. Format output in clean Markdown with clear sections.
6. Be specific, actionable, and concise.
"""

        user_prompt = f"""
CUSTOMER PROFILE:
{json.dumps(profile, indent=2)}

CHURN RISK ASSESSMENT:
{state['risk_summary']}

RETRIEVED RETENTION STRATEGIES (use ONLY these as your source):
{context_text}

USER QUERY / CONTEXT:
{state.get('query', 'Generate a full retention strategy report.')}

---
Generate a retention strategy report with EXACTLY these sections:

## 1. Executive Risk Summary
(2-3 sentences on the customer's risk profile)

## 2. Key Risk Factors
(Bullet list of top 3-5 churn drivers for this customer)

## 3. Recommended Retention Actions
(Numbered list of 3-5 specific, prioritized actions with rationale)

## 4. Expected Outcomes
(Brief description of expected impact if recommendations are followed)

## 5. Sources & References
(List the strategy documents used)

## 6. Ethical & Business Disclaimer
(Standard ethical AI disclosure)
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,   # Low temperature = less hallucination
            max_tokens=1200,
        )

        report = response.choices[0].message.content

        return {**state, "retention_report": report}

    except Exception as e:
        return {
            **state,
            "retention_report": _fallback_report(state),
            "error": f"LLM generation error: {str(e)} — showing rule-based fallback report.",
        }


def _fallback_report(state: AgentState) -> str:
    """
    Rule-based fallback report when Groq API is unavailable.
    Ensures the app degrades gracefully.
    """
    profile = state["customer_profile"]
    prob = state["churn_probability"]
    tier = state.get("risk_tier", "UNKNOWN")
    docs = state.get("retrieved_docs", [])

    actions = []
    if profile.get("Support Calls", 0) >= 5:
        actions.append("Assign a dedicated support specialist and conduct a root-cause resolution call.")
    if profile.get("Payment Delay", 0) >= 20:
        actions.append("Contact billing team to offer a flexible payment restructuring plan.")
    if profile.get("Usage Frequency", 0) <= 5:
        actions.append("Send a personalized re-engagement email with usage tips and a free training session.")
    if profile.get("Tenure", 0) <= 6:
        actions.append("Schedule an onboarding refresh call and demonstrate key value features.")
    if prob >= 0.70:
        actions.append("Offer a time-limited retention package (discount or plan upgrade) within 24 hours.")
    if not actions:
        actions.append("Enroll customer in a proactive loyalty program and schedule a check-in call.")

    action_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
    doc_refs = "\n".join(f"- {d['title']}" for d in docs) if docs else "- Internal retention knowledge base"

    return f"""## 1. Executive Risk Summary
This customer has a **{tier}** churn risk with a predicted probability of **{prob:.1%}**.
Immediate action is {'strongly ' if prob >= 0.70 else ''}recommended to prevent churn.

## 2. Key Risk Factors
{_get_risk_factors(profile)}

## 3. Recommended Retention Actions
{action_text}

## 4. Expected Outcomes
Following these recommendations is expected to reduce churn probability by 20–40% within 30 days,
depending on customer responsiveness and execution quality.

## 5. Sources & References
{doc_refs}

## 6. Ethical & Business Disclaimer
> This report is generated by an AI system and is probabilistic in nature. All recommendations
> should be validated by a human customer success manager before execution. Customer data is
> used solely for retention purposes in compliance with applicable data privacy regulations.
> Discount offers and interventions must be applied equitably across similar customer segments.
"""


def _get_risk_factors(profile: dict) -> str:
    factors = []
    if profile.get("Support Calls", 0) >= 5:
        factors.append(f"- High support call volume ({profile['Support Calls']} calls)")
    if profile.get("Payment Delay", 0) >= 20:
        factors.append(f"- Payment delays ({profile['Payment Delay']} days)")
    if profile.get("Usage Frequency", 0) <= 5:
        factors.append(f"- Low usage frequency ({profile['Usage Frequency']} sessions)")
    if profile.get("Tenure", 0) <= 6:
        factors.append(f"- Short customer tenure ({profile['Tenure']} months)")
    if profile.get("Last Interaction", 0) >= 25:
        factors.append(f"- Low recent interaction ({profile['Last Interaction']} days)")
    if not factors:
        factors.append("- No critical behavioral risk signals — monitor proactively")
    return "\n".join(factors)


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

def build_agent_graph():
    """Build and compile the LangGraph retention agent."""
    graph = StateGraph(AgentState)

    graph.add_node("analyze_risk", analyze_risk_node)
    graph.add_node("retrieve_strategies", retrieve_strategies_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "analyze_risk")
    graph.add_edge("analyze_risk", "retrieve_strategies")
    graph.add_edge("retrieve_strategies", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# Singleton compiled graph
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent


def run_retention_agent(
    customer_profile: dict,
    churn_probability: float,
    query: str = "",
) -> AgentState:
    """
    Main entry point to run the retention agent.
    Returns the final state with retention_report and all intermediate outputs.
    """
    agent = get_agent()
    initial_state: AgentState = {
        "customer_profile": customer_profile,
        "churn_probability": churn_probability,
        "query": query,
        "risk_tier": "",
        "risk_summary": "",
        "retrieved_docs": [],
        "retention_report": "",
        "error": "",
    }
    result = agent.invoke(initial_state)
    return result
