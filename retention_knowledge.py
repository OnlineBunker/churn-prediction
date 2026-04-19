"""
Retention Strategy Knowledge Base
This module contains the curated retention strategy documents used for RAG retrieval.
"""

RETENTION_DOCS = [
    # High Churn Risk - General
    {
        "id": "doc_001",
        "title": "High-Risk Customer Immediate Intervention",
        "content": (
            "Customers with churn probability above 70% require immediate intervention. "
            "Best practice: assign a dedicated customer success manager within 24 hours. "
            "Offer a personalized retention package including a discount (10-20%), service upgrade, "
            "or extended contract benefit. Personal outreach via phone call is significantly more "
            "effective than email for high-risk segments. Acknowledge any past service issues proactively."
        ),
        "tags": ["high_risk", "immediate", "intervention"],
    },
    {
        "id": "doc_002",
        "title": "Proactive Loyalty Programs",
        "content": (
            "Loyalty programs reduce churn by 15-25% on average. "
            "Introduce tiered reward systems where customers earn points for tenure and spend. "
            "Milestone rewards (e.g., 1-year anniversary gift) create emotional loyalty. "
            "Early renewal incentives (discount if renewing 60 days before contract end) are "
            "highly effective for contract-based customers."
        ),
        "tags": ["loyalty", "program", "retention", "medium_risk"],
    },
    {
        "id": "doc_003",
        "title": "Support Experience Improvement",
        "content": (
            "High support call volume is a strong churn predictor. "
            "Customers with 5+ support calls in a quarter have 3x higher churn risk. "
            "Strategies: implement proactive support outreach after 2nd ticket, offer dedicated "
            "support lines for at-risk customers, conduct root-cause resolution calls. "
            "Follow-up satisfaction surveys after resolution reduce repeat tickets by 30%."
        ),
        "tags": ["support", "high_support_calls", "service_quality"],
    },
    {
        "id": "doc_004",
        "title": "Payment Delay Recovery Strategy",
        "content": (
            "Payment delays exceeding 30 days increase churn probability by 40%. "
            "Recommended actions: send a friendly payment reminder at day 7, offer flexible "
            "payment plan restructuring at day 15, assign a billing specialist at day 25. "
            "Avoid aggressive collection tactics — empathetic outreach retains 60% of delayed payers. "
            "Auto-pay enrollment discounts can prevent future payment delays."
        ),
        "tags": ["payment", "delay", "billing", "financial_risk"],
    },
    {
        "id": "doc_005",
        "title": "Re-engagement for Low Usage Customers",
        "content": (
            "Customers with usage frequency below 20% of their plan are prime churn risks. "
            "Re-engagement playbook: send personalized usage tips, offer a free training session "
            "or onboarding refresh, highlight unused features relevant to their profile. "
            "Usage-based nudges ('You haven't used Feature X — here's how it helps you') "
            "show 35% higher open rates than generic campaigns."
        ),
        "tags": ["low_usage", "re-engagement", "product_adoption"],
    },
    {
        "id": "doc_006",
        "title": "New Customer Onboarding & Early Tenure Retention",
        "content": (
            "The first 90 days are critical — 40% of churned customers leave within this window. "
            "Best practices: structured onboarding sequence with check-in calls at day 7, 30, and 60. "
            "Assign an onboarding specialist for enterprise accounts. "
            "Send 'quick win' tutorials in the first week to demonstrate immediate value. "
            "Welcome surveys at day 14 catch early dissatisfaction before it escalates."
        ),
        "tags": ["new_customer", "onboarding", "early_tenure"],
    },
    {
        "id": "doc_007",
        "title": "Contract Renewal Negotiation",
        "content": (
            "Customers approaching contract end have a 50% higher churn rate. "
            "Begin renewal conversations 90 days before contract expiry. "
            "Offer multi-year contract discounts (5% for 2-year, 12% for 3-year). "
            "Provide a personalized ROI summary showing value delivered. "
            "Executive-level engagement for high-value accounts increases renewal rate by 25%."
        ),
        "tags": ["contract", "renewal", "negotiation", "long_tenure"],
    },
    {
        "id": "doc_008",
        "title": "Personalized Discount & Offer Strategy",
        "content": (
            "Blanket discounts are ineffective and erode margins. "
            "Segment-specific offers work best: for price-sensitive customers, offer a 15% discount; "
            "for feature-seekers, upgrade their plan at same price; "
            "for support-frustrated customers, offer a service credit. "
            "Discounts should have a clear expiry (e.g., 48-hour offer window) to drive urgency. "
            "Track offer acceptance rates and adjust by segment."
        ),
        "tags": ["discount", "offer", "personalization", "pricing"],
    },
    {
        "id": "doc_009",
        "title": "Senior Customer Retention",
        "content": (
            "Older customers (age 55+) value reliability and personal relationships over features. "
            "Phone outreach is 3x more effective than digital campaigns for this segment. "
            "Simplify account management interfaces and provide dedicated support lines. "
            "Offer annual plan reviews with a human advisor. "
            "Emphasize trust, stability, and continuity in all communications."
        ),
        "tags": ["senior", "age", "demographics", "personalization"],
    },
    {
        "id": "doc_010",
        "title": "Win-Back Strategy for Near-Churned Customers",
        "content": (
            "For customers who have submitted cancellation requests or downgraded, "
            "a structured win-back program recovers 15-30% of at-risk accounts. "
            "Step 1: Senior retention specialist calls within 2 hours of cancellation intent. "
            "Step 2: Understand root cause — do NOT immediately offer discounts. "
            "Step 3: Offer a customized resolution package based on stated reason. "
            "Step 4: If declined, send a follow-up exit survey and 30-day win-back email sequence."
        ),
        "tags": ["win_back", "cancellation", "critical", "very_high_risk"],
    },
    {
        "id": "doc_011",
        "title": "Subscription Type-Based Retention",
        "content": (
            "Basic plan customers churn at 2x the rate of premium subscribers. "
            "Strategy: for basic plan churners, offer a time-limited premium trial (14-30 days). "
            "Demonstrate premium features they would benefit from based on usage data. "
            "Post-trial conversion offers of 25% off first premium year show strong results. "
            "Premium customers should receive exclusive early access to new features."
        ),
        "tags": ["subscription", "plan_upgrade", "upsell", "retention"],
    },
    {
        "id": "doc_012",
        "title": "Ethical AI Disclosure in Retention Recommendations",
        "content": (
            "AI-driven retention strategies must be deployed ethically. "
            "Customers should not be manipulated using psychological pressure tactics. "
            "Discount offers must be genuine and available to all customers in similar situations. "
            "Personalized outreach must respect communication preferences (opt-outs). "
            "All AI predictions are probabilistic — human judgment must validate recommendations. "
            "Customer data used for churn modeling must comply with data privacy regulations (GDPR, CCPA)."
        ),
        "tags": ["ethics", "ai_disclosure", "compliance", "privacy"],
    },
]
