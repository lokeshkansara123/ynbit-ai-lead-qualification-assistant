import re
from datetime import datetime
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Cynbit Inquiry Intelligence Assistant", page_icon="🧠", layout="wide")

# Demonstration only: training examples are synthetic and hand-authored.
# Do not treat the model's output as validated business performance.

TRAINING_EXAMPLES = [
    ("We need an AI chatbot and LLM assistant for customer support", "AI / LLM Solution"),
    ("Build a RAG assistant to search internal documents", "AI / LLM Solution"),
    ("Automate manual data entry using an AI agent", "AI / LLM Solution"),
    ("Create a virtual assistant with prompt engineering", "AI / LLM Solution"),
    ("Integrate a language model into our existing SaaS product", "AI / LLM Solution"),
    ("We need document extraction using AI and OCR", "AI / LLM Solution"),
    ("Develop a website and e-commerce web application", "Web / Mobile Development"),
    ("Build a mobile app for customers to place orders", "Web / Mobile Development"),
    ("We need a new frontend and backend for our platform", "Web / Mobile Development"),
    ("Create a customer portal and responsive website", "Web / Mobile Development"),
    ("Develop an online store with product pages and checkout", "Web / Mobile Development"),
    ("Build a booking application for mobile and web", "Web / Mobile Development"),
    ("Migrate our application to AWS cloud infrastructure", "Cloud / DevOps"),
    ("Set up Docker deployment and CI CD pipeline", "Cloud / DevOps"),
    ("We need Azure cloud migration and monitoring", "Cloud / DevOps"),
    ("Improve deployment reliability and server scaling", "Cloud / DevOps"),
    ("Configure Kubernetes and cloud security", "Cloud / DevOps"),
    ("Move our legacy software to cloud hosting", "Cloud / DevOps"),
    ("Integrate Salesforce CRM and automate sales follow ups", "CRM / Sales Automation"),
    ("Build a lead management workflow for our sales team", "CRM / Sales Automation"),
    ("Connect HubSpot with our existing business systems", "CRM / Sales Automation"),
    ("Automate CRM updates and sales pipeline tracking", "CRM / Sales Automation"),
    ("We need customer relationship management automation", "CRM / Sales Automation"),
    ("Create automated lead nurturing and follow up reminders", "CRM / Sales Automation"),
    ("Build a dashboard for sales and marketing analytics", "Data / Analytics"),
    ("Analyze business data and create weekly reports", "Data / Analytics"),
    ("We need a Power BI dashboard and data pipeline", "Data / Analytics"),
    ("Forecast demand from historical sales data", "Data / Analytics"),
    ("Combine multiple data sources for KPI reporting", "Data / Analytics"),
    ("Create an analytics dashboard for business performance", "Data / Analytics"),
    ("Improve our SEO and digital marketing campaigns", "Digital Marketing / SEO"),
    ("We need Google Ads and Meta Ads campaign optimization", "Digital Marketing / SEO"),
    ("Plan social media marketing and search optimization", "Digital Marketing / SEO"),
    ("Audit our website SEO and improve organic traffic", "Digital Marketing / SEO"),
    ("Track marketing campaign performance and conversions", "Digital Marketing / SEO"),
    ("Help with paid advertising and lead generation", "Digital Marketing / SEO"),
]

CATEGORY_KEYWORDS = {
    "AI / LLM Solution": ["llm", "rag", "ai agent", "ai chatbot", "machine learning", "generative ai", "automation", "chatbot", "document extraction", "virtual assistant"],
    "Web / Mobile Development": ["website", "web app", "mobile app", "frontend", "backend", "e-commerce", "ecommerce", "customer portal", "online store"],
    "Cloud / DevOps": ["aws", "azure", "cloud", "docker", "kubernetes", "deployment", "devops", "migration", "ci cd"],
    "CRM / Sales Automation": ["crm", "salesforce", "lead management", "sales automation", "pipeline", "customer relationship", "hubspot", "lead nurturing"],
    "Data / Analytics": ["dashboard", "analytics", "data pipeline", "power bi", "reporting", "forecasting", "data science", "kpi"],
    "Digital Marketing / SEO": ["seo", "google ads", "meta ads", "digital marketing", "campaign", "social media", "organic traffic"],
}

SERVICE_MAP = {
    "AI / LLM Solution": "Run a discovery workshop, identify a high-value workflow, and scope a grounded AI/LLM proof of concept.",
    "Web / Mobile Development": "Confirm user journeys and integrations, then prepare a phased web/mobile development scope.",
    "Cloud / DevOps": "Review the current environment, availability/security needs, deployment pipeline, and cloud migration constraints.",
    "CRM / Sales Automation": "Map the current sales process, CRM platform, lead lifecycle, and follow-up automation requirements.",
    "Data / Analytics": "Confirm data sources, KPI definitions, data quality, and the dashboard/reporting audience.",
    "Digital Marketing / SEO": "Review baseline channel metrics, conversion goals, tracking setup, and optimization priorities.",
    "General / Needs Discovery": "Arrange a discovery call to clarify the business goal, scope, integrations, timeline, and budget."
}

@st.cache_resource
def get_classifier():
    texts = [x[0] for x in TRAINING_EXAMPLES]
    labels = [x[1] for x in TRAINING_EXAMPLES]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(texts, labels)
    return model

def extract_requirements(text):
    lower = text.lower()
    groups = {
        "AI / chatbot": ["ai", "chatbot", "llm", "agent", "rag", "language model"],
        "CRM integration": ["crm", "salesforce", "hubspot", "integration"],
        "Web / mobile": ["website", "web app", "mobile app", "e-commerce", "ecommerce", "frontend", "backend"],
        "Cloud / deployment": ["aws", "azure", "cloud", "deploy", "docker", "kubernetes"],
        "Analytics / reporting": ["dashboard", "analytics", "report", "forecast", "power bi", "kpi"],
        "Multilingual support": ["multilingual", "multiple languages", "hindi", "language support"],
        "Automation": ["automate", "automation", "manual process", "workflow", "follow up"],
        "Security / privacy": ["security", "secure", "privacy", "compliance", "access control"],
    }
    found = [label for label, words in groups.items() if any(word in lower for word in words)]
    return found or ["Needs clarification"]

def score_lead(text, budget, timeline, contact_provided):
    # Transparent heuristic score; not trained on conversion outcomes.
    lower = text.lower()
    score, reasons = 25, ["Baseline triage points (+25)"]
    if any(k in lower for k in ["integration", "integrate", "existing system", "crm", "api"]):
        score += 15; reasons.append("Integration detail mentioned (+15)")
    if any(k in lower for k in ["business", "customers", "users", "team", "production", "company"]):
        score += 10; reasons.append("Business context mentioned (+10)")
    if budget == "Defined":
        score += 20; reasons.append("Budget status marked defined (+20)")
    if timeline == "Defined":
        score += 15; reasons.append("Timeline status marked defined (+15)")
    if contact_provided:
        score += 10; reasons.append("Contact email provided (+10)")
    if len(text.split()) >= 35:
        score += 5; reasons.append("Detailed inquiry (+5)")
    score = min(score, 100)
    band = "High" if score >= 70 else ("Medium" if score >= 45 else "Low")
    return score, band, reasons

def clarification_questions(text, budget, timeline):
    lower = text.lower()
    qs = []
    if budget == "Unknown":
        qs.append("What budget range has been allocated?")
    if timeline == "Unknown" or not any(x in lower for x in ["week", "month", "date", "deadline", "timeline"]):
        qs.append("Is there a target launch date or preferred delivery timeline?")
    if not any(x in lower for x in ["crm", "salesforce", "hubspot", "api", "integration"]):
        qs.append("Which existing systems or tools must the solution integrate with?")
    if not any(x in lower for x in ["users", "customers", "traffic", "employees", "team"]):
        qs.append("Who will use the solution and what usage volume is expected?")
    if not any(x in lower for x in ["success", "kpi", "reduce", "increase", "save time", "conversion", "goal"]):
        qs.append("What measurable business outcome would define success?")
    return qs[:5]

def proposal_draft(category, requirements, inquiry, budget, timeline):
    req = ", ".join(requirements)
    return f"""INITIAL PROJECT DISCUSSION DRAFT — NOT A QUOTATION

1. Client objective
{inquiry.strip()}

2. Suggested service area
{category}

3. Suggested discovery approach
{SERVICE_MAP.get(category, SERVICE_MAP["General / Needs Discovery"])}

4. Candidate requirements identified
{req}

5. Items to confirm
- Budget status: {budget}
- Timeline status: {timeline}
- Required integrations, data access, security, ownership, and acceptance criteria.
- Any final scope, delivery date, and commercial estimate must be confirmed by an authorized human reviewer.

6. Recommended next step
Schedule a discovery call, confirm requirements, and prepare a reviewed scope document.

This draft is generated by a demonstration tool using the inquiry text and fixed templates. It is not an approved Cynbit proposal, price quote, or delivery commitment.
"""

st.title("🧠 Cynbit Inquiry Intelligence & Proposal Assistant")
st.caption("AI/ML Engineer Intern PoC • Synthetic training examples • Not connected to Cynbit systems")
st.warning("The classifier is trained only on small, hand-written synthetic examples. Its predictions are for demonstration, not production decisions. The lead priority score is a separate transparent heuristic, not a conversion probability.")

with st.sidebar:
    st.header("Optional lead context")
    contact_name = st.text_input("Contact name", placeholder="Optional")
    company = st.text_input("Company", placeholder="Optional")
    email = st.text_input("Email", placeholder="Optional")
    budget = st.selectbox("Budget status", ["Unknown", "Defined"])
    timeline = st.selectbox("Timeline status", ["Unknown", "Defined"])
    st.caption("Use fictional examples. Avoid confidential customer information.")

default_inquiry = ("We need an AI chatbot for our e-commerce website. It should answer customer questions, "
                   "support multiple languages, and integrate with our existing CRM. Our team hopes to launch "
                   "in six weeks and wants to reduce repetitive customer support work.")
inquiry = st.text_area("Project inquiry", value=default_inquiry, height=160)
if st.button("Analyze inquiry", type="primary", use_container_width=True):
    if len(inquiry.strip()) < 15:
        st.error("Please enter a more detailed inquiry.")
    else:
        model = get_classifier()
        category = model.predict([inquiry])[0]
        probabilities = model.predict_proba([inquiry])[0]
        confidence = float(max(probabilities))
        requirements = extract_requirements(inquiry)
        score, band, reasons = score_lead(inquiry, budget, timeline, bool(email.strip()))
        questions = clarification_questions(inquiry, budget, timeline)
        draft = proposal_draft(category, requirements, inquiry, budget, timeline)

        st.subheader("Analysis")
        a, b, c = st.columns(3)
        a.metric("Suggested category", category)
        b.metric("Model confidence (demo)", f"{confidence:.0%}")
        c.metric("Rule-based priority", f"{score}/100 · {band}")
        st.caption("Confidence is calculated from a model trained on synthetic examples and is not calibrated or validated.")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Detected requirements")
            for r in requirements: st.write("• " + r)
            st.markdown("#### Suggested service approach")
            st.write(SERVICE_MAP.get(category, SERVICE_MAP["General / Needs Discovery"]))
        with right:
            st.markdown("#### Priority score explanation")
            for reason in reasons: st.write("• " + reason)
            st.markdown("#### Clarification questions")
            for q in questions: st.write("• " + q)

        st.markdown("#### Proposal draft")
        st.text_area("Review and edit before use", value=draft, height=320, key="draft_area")
        st.download_button("Download proposal draft (.txt)", data=draft, file_name="proposal_draft.txt", mime="text/plain")
        record = {
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "contact_name": contact_name, "company": company, "email": email,
            "inquiry": inquiry, "suggested_category": category,
            "synthetic_model_confidence": round(confidence, 4),
            "requirements": "; ".join(requirements),
            "rule_based_priority_score": score, "priority_band": band,
            "budget_status": budget, "timeline_status": timeline,
            "clarification_questions": " | ".join(questions),
        }
        st.download_button("Download analysis (.csv)", data=pd.DataFrame([record]).to_csv(index=False),
                           file_name="lead_analysis.csv", mime="text/csv")

with st.expander("Synthetic test cases"):
    st.markdown("""
    - **Cloud:** Our team wants to migrate a legacy application to AWS, use Docker, and improve deployment reliability.
    - **CRM:** We need Salesforce integration, lead management, and automated sales follow-ups.
    - **Analytics:** We need a dashboard to combine sales data and campaign performance for weekly reporting.
    - **Marketing:** We want to improve SEO, Google Ads performance, and campaign conversion tracking.

    These examples are fictional and do not represent actual Cynbit clients.
    """)
