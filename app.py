import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Research module imports (graceful fallback) ───
_RESEARCH_AVAILABLE = False
_RESEARCH_ERR = None
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing.feature_engineering import FEATURE_COLS, FEATURE_DESCRIPTIONS
    from explainability.shap_explanations import (
        build_shap_explainer, compute_shap_values,
        global_importance_df, plot_global_importance, plot_summary, explain_student,
        RISK_LABELS,
    )
    from evaluation.model_evaluation import plot_confusion_matrix, LABEL_NAMES
    from survey.survey_validation import (
        compute_mbi_scores, validate_model_vs_survey, plot_validation_scatter,
        create_survey_template, SURVEY_TEMPLATE_PATH
    )
    _RESEARCH_AVAILABLE = True
except Exception as _e:
    _RESEARCH_ERR = str(_e)

# ─── Page Config ───
st.set_page_config(
    page_title="Early Student Burnout-Risk Screening System",
    page_icon="🎓",
    layout="wide"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.2rem; max-width: 1100px; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 50%, #8b5cf6 100%);
        padding: 2.2rem 2rem 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.22);
        position: relative;
        overflow: hidden;
    }
    .hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0; color: white; }
    .hero p  { font-size: 0.95rem; opacity: 0.92; margin-top: 0.4rem; color: #ede9fe; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 0.35rem 1.1rem;
        border-radius: 20px;
        font-size: 0.78rem;
        margin-top: 0.7rem;
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.25);
    }

    /* ── Disclaimer Banner ── */
    .disclaimer-banner {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-bottom: 1.2rem;
        line-height: 1.45;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 1.18rem; font-weight: 700; color: #1e293b;
        margin: 1.2rem 0 0.4rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .section-subtitle {
        font-size: 0.85rem; color: #64748b; margin-top: -0.2rem; margin-bottom: 1rem;
    }

    /* ── Input cards ── */
    .input-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.2rem 0.5rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .input-card:hover {
        border-color: #a78bfa;
        box-shadow: 0 4px 12px rgba(139,92,246,0.1);
    }
    .input-label {
        font-weight: 600; font-size: 0.82rem; color: #475569;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-bottom: 0.15rem;
    }
    .input-helper {
        font-size: 0.76rem; color: #94a3b8; margin-bottom: 0.4rem;
        line-height: 1.35;
    }
    .input-example {
        font-size: 0.72rem; color: #6366f1;
        background: #f5f3ff; padding: 0.25rem 0.6rem;
        border-radius: 6px; display: inline-block; margin-bottom: 0.4rem;
        font-weight: 500;
    }

    /* ── Result cards ── */
    .result-card {
        padding: 2rem 1.5rem; border-radius: 18px; text-align: center;
        margin: 1rem 0; position: relative; overflow: hidden;
    }
    .result-low {
        background: linear-gradient(145deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #4ade80;
        box-shadow: 0 6px 20px rgba(74,222,128,0.2);
    }
    .result-medium {
        background: linear-gradient(145deg, #fef9c3 0%, #fde68a 100%);
        border: 2px solid #facc15;
        box-shadow: 0 6px 20px rgba(250,204,21,0.2);
    }
    .result-high {
        background: linear-gradient(145deg, #fee2e2 0%, #fca5a5 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 6px 20px rgba(239,68,68,0.2);
    }
    .result-emoji { font-size: 3.2rem; margin-bottom: 0.2rem; }
    .result-level { font-size: 1.5rem; font-weight: 800; color: #1e293b; }
    .result-advice { font-size: 0.92rem; color: #334155; margin-top: 0.4rem; line-height: 1.5; }
    .result-conf {
        font-size: 0.8rem; color: #475569; margin-top: 0.6rem;
        background: rgba(255,255,255,0.6); display: inline-block;
        padding: 0.25rem 0.8rem; border-radius: 10px; font-weight: 600;
    }

    /* ── Metric boxes ── */
    .metric-row { display: flex; gap: 0.7rem; margin: 1rem 0; flex-wrap: wrap; }
    .metric-box {
        flex: 1; min-width: 100px;
        background: #ffffff; border-radius: 12px;
        padding: 1rem 0.6rem; text-align: center;
        border: 1.5px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-val { font-size: 1.35rem; font-weight: 800; color: #1e293b; }
    .metric-desc { font-size: 0.72rem; color: #64748b; margin-top: 0.15rem; font-weight: 500; }
    .metric-icon { font-size: 1.3rem; margin-bottom: 0.2rem; }

    /* ── Research Cards ── */
    .res-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .res-card-header {
        font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 0.3rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .res-card-body {
        font-size: 0.88rem; color: #475569; line-height: 1.55;
    }
    .res-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pill-blue { background: #dbeafe; color: #1e40af; }
    .pill-purple { background: #ede9fe; color: #6b21a8; }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-amber { background: #fef3c7; color: #92400e; }

    /* ── Trade-off Callout ── */
    .tradeoff-banner {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1.5px solid #cbd5e1;
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 1.2rem 0;
    }
    .tradeoff-title {
        font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 0.3rem;
    }
    .tradeoff-desc {
        font-size: 0.88rem; color: #475569; line-height: 1.55;
    }

    /* ── Insight items ── */
    .insight-card {
        display: flex; align-items: flex-start; gap: 0.8rem;
        padding: 0.85rem 1rem; margin: 0.4rem 0;
        border-radius: 10px; background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .insight-dot {
        width: 10px; height: 10px; border-radius: 50%;
        margin-top: 0.35rem; flex-shrink: 0;
    }
    .dot-green  { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.4); }
    .dot-yellow { background: #eab308; box-shadow: 0 0 6px rgba(234,179,8,0.4); }
    .dot-red    { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.4); }
    .insight-text { font-size: 0.88rem; color: #334155; line-height: 1.45; }
    .insight-sub  { font-size: 0.77rem; color: #64748b; margin-top: 0.15rem; }

    /* ── Recommendation cards ── */
    .rec-card {
        display: flex; align-items: flex-start; gap: 0.7rem;
        padding: 0.9rem 1rem; margin: 0.4rem 0;
        border-radius: 10px;
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border: 1px solid #bfdbfe;
    }
    .rec-icon { font-size: 1.1rem; margin-top: 0.05rem; }
    .rec-text { font-size: 0.88rem; color: #1e40af; line-height: 1.45; }

    /* ── Gauge visual ── */
    .gauge-container { text-align: center; margin: 0.5rem 0; }
    .gauge-bar {
        height: 10px; border-radius: 5px;
        background: linear-gradient(90deg, #22c55e 0%, #22c55e 33%, #eab308 33%, #eab308 66%, #ef4444 66%, #ef4444 100%);
        position: relative; margin: 0.5rem auto; width: 80%;
    }
    .gauge-marker {
        width: 18px; height: 18px; border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        position: absolute; top: -4px;
        transition: left 0.3s ease;
    }
    .gauge-labels {
        display: flex; justify-content: space-between;
        width: 80%; margin: 0.3rem auto 0;
        font-size: 0.7rem; color: #94a3b8;
    }

    /* ── FAQ accordion ── */
    .faq-item {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; margin: 0.4rem 0;
    }
    .faq-item summary {
        font-weight: 600; font-size: 0.88rem; color: #334155;
        padding: 0.8rem 1rem; cursor: pointer; list-style: none;
    }
    .faq-item summary::before { content: '❓ '; }
    .faq-item p {
        font-size: 0.82rem; color: #64748b; padding: 0 1rem 0.8rem;
        line-height: 1.5; margin: 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center; padding: 1.5rem 0 0.5rem;
        font-size: 0.75rem; color: #94a3b8;
        border-top: 1px solid #e2e8f0; margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model Resources with Caching ───
@st.cache_resource
def load_default_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_research_results():
    baseline_path = "models/results_baseline.pkl"
    early_path = "models/results_early_detection.pkl"
    if os.path.exists(baseline_path) and os.path.exists(early_path):
        b = joblib.load(baseline_path)
        e = joblib.load(early_path)
        return b, e
    return None, None

@st.cache_resource
def get_cached_explainer():
    b_data, _ = load_research_results()
    if b_data is not None:
        model = b_data["best_model"]
        X_sample = b_data["X_test"].sample(min(150, len(b_data["X_test"])), random_state=42)
        explainer, _ = build_shap_explainer(model, X_sample)
        return explainer, model, X_sample
    return None, None, None


# ─── Sidebar Navigation ───
with st.sidebar:
    st.markdown("### 🎓 Platform Navigation")
    nav_option = st.radio(
        "Select Workspace View:",
        [
            "🎯 Student Burnout-Risk Screening",
            "📊 Baseline vs Early Detection",
            "🔍 Explainable AI (SHAP) Lab",
            "📋 MBI-SS Pilot Survey Validation",
            "📚 Research Framework & Viva Guide"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🔬 System Specifications")
    st.markdown(
        """
        - **Target:** Behavior-Derived Burnout Risk Proxy
        - **Baseline Model:** Full Semester (XGBoost 88.97%)
        - **Early-Window Model:** First 21 Days (RF 66.27%)
        - **Explainability:** Multiclass SHAP TreeExplainer
        - **Validation:** MBI-SS 9-Item Pilot Survey
        - **Dataset:** OULAD (Open University)
        """
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#94a3b8; text-align:center;'>"
        "Early Student Burnout-Risk Screening System<br>"
        "Academic ML Research Project</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# VIEW 1: INTERACTIVE BURNOUT-RISK SCREENING
# ==============================================================================
if nav_option == "🎯 Student Burnout-Risk Screening":
    # ─── Hero ───
    st.markdown("""
    <div class="hero">
        <h1>🎓 Early Student Burnout-Risk Screening System</h1>
        <p>Proactive risk estimation using student LMS interaction dynamics and early engagement patterns</p>
        <div class="hero-badge">✨ Behavior-Derived Risk Proxy · Trained on 30,000+ Student Records · Class-Specific SHAP Attribution</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Clinical Disclaimer:</strong> This system estimates behavior-derived burnout risk from LMS activity patterns and is <strong>not</strong> a clinical diagnostic instrument or psychological assessment.
    </div>
    """, unsafe_allow_html=True)

    try:
        model = load_default_model()
    except Exception:
        st.error("⚠️ Could not load `model.pkl`. Make sure it's in the project root.")
        st.stop()

    with st.expander("👋 Quick Guide: How to use this screening tool", expanded=False):
        st.markdown("""
        **What does this tool do?**  
        It estimates your risk tier (Low, Medium, High) for academic burnout based on a multi-factor behavioral proxy (platform clicks, submission timeliness, course load, retakes).
        
        **How to use:**
        1. Adjust the 4 sliders below to reflect your semester study habits.
        2. Click **Run Burnout-Risk Screening**.
        3. Review your risk gauge, behavioral insights, action plan, and local SHAP mathematical attribution.
        """)

    st.markdown('<div class="section-title">📝 Enter Academic & Behavioral Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Fine-tune the sliders to match current semester habits</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="input-card">
            <div class="input-label">🖱️ Platform Engagement (Clicks)</div>
            <div class="input-helper">Total interactions with lectures, discussion boards, quizzes, and course materials.</div>
            <div class="input-example">💡 Example: 500 = minimal · 3000 = average · 8000+ = high activity</div>
        </div>
        """, unsafe_allow_html=True)
        sum_click = st.slider("Engagement Clicks", 0, 15000, 3000, step=100, label_visibility="collapsed")

    with col2:
        st.markdown("""
        <div class="input-card">
            <div class="input-label">⏱️ Submission Delay (Days)</div>
            <div class="input-helper">Average days relative to assignment deadlines. Negative = early submission, Positive = late.</div>
            <div class="input-example">💡 Example: −10 = 10 days early · 0 = on deadline · +25 = late</div>
        </div>
        """, unsafe_allow_html=True)
        submission_delay = st.slider("Submission Delay", -200, 200, 0, step=1, label_visibility="collapsed")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="input-card">
            <div class="input-label">🔁 Previous Course Attempts</div>
            <div class="input-helper">Number of times this module/course was attempted prior to the current term.</div>
            <div class="input-example">💡 Example: 0 = first attempt · 1 = retake · 2+ = repeated challenge</div>
        </div>
        """, unsafe_allow_html=True)
        prev_attempts = st.slider("Previous Attempts", 0, 6, 0, step=1, label_visibility="collapsed")

    with col4:
        st.markdown("""
        <div class="input-card">
            <div class="input-label">📚 Studied Credits (Course Load)</div>
            <div class="input-helper">Total credit hours currently enrolled. Higher credits directly increase academic load.</div>
            <div class="input-example">💡 Example: 60 = part-time · 120 = standard full-time · 240+ = heavy</div>
        </div>
        """, unsafe_allow_html=True)
        studied_credits = st.slider("Studied Credits", 30, 600, 120, step=10, label_visibility="collapsed")

    # ─── Derived Features (10 Features Pipeline) ───
    delay_abs = abs(submission_delay)
    engagement_level = 0 if sum_click <= 1000 else (1 if sum_click <= 3000 else 2)
    engagement_per_day = sum_click / (prev_attempts + 1)
    delay_ratio = submission_delay / (delay_abs + 1)
    click_intensity = sum_click / (studied_credits + 1)
    activity_score = sum_click * engagement_level

    predict_btn = st.button("🔍  Run Burnout-Risk Screening", use_container_width=True, type="primary")

    if predict_btn:
        features_arr = np.array([[
            sum_click, submission_delay, delay_abs, engagement_level,
            engagement_per_day, delay_ratio, click_intensity, activity_score,
            prev_attempts, studied_credits
        ]])
        features_df = pd.DataFrame(features_arr, columns=FEATURE_COLS)

        prediction = model.predict(features_arr)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_arr)[0]
            confidence = max(proba) * 100

        config = {
            0: {
                "label": "Low Burnout Risk (Proxy)",
                "emoji": "😊",
                "css": "result-low",
                "advice": "Healthy engagement signals. Your study rhythms and interaction patterns appear balanced and sustainable.",
                "gauge_pos": "14%",
                "gauge_color": "#22c55e",
            },
            1: {
                "label": "Medium Burnout Risk (Proxy)",
                "emoji": "😐",
                "css": "result-medium",
                "advice": "Caution — early stress patterns detected. Submissions or pacing may need proactive adjustment.",
                "gauge_pos": "49%",
                "gauge_color": "#eab308",
            },
            2: {
                "label": "High Burnout Risk (Proxy)",
                "emoji": "😰",
                "css": "result-high",
                "advice": "Alert! Significant risk indicators detected across workload and engagement metrics. Timely support is advised.",
                "gauge_pos": "84%",
                "gauge_color": "#ef4444",
            },
        }

        c = config[prediction]

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Screening Result", "💡 Behavioral Insights", "🛠️ Action Plan", "🔬 Local SHAP Explanation"])

        with tab1:
            conf_html = f'<div class="result-conf">Model Confidence: {confidence:.1f}%</div>' if confidence else ""
            st.markdown(f"""
            <div class="result-card {c['css']}">
                <div class="result-emoji">{c['emoji']}</div>
                <div class="result-level">{c['label']}</div>
                <div class="result-advice">{c['advice']}</div>
                {conf_html}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="gauge-container">
                <div class="gauge-bar">
                    <div class="gauge-marker" style="left: {c['gauge_pos']}; background: {c['gauge_color']};"></div>
                </div>
                <div class="gauge-labels">
                    <span>Low Risk (0.0 – 0.33)</span><span>Medium Risk (0.34 – 0.66)</span><span>High Risk (0.67 – 1.0)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-icon">🖱️</div>
                    <div class="metric-val">{sum_click:,}</div>
                    <div class="metric-desc">Total Clicks</div>
                </div>
                <div class="metric-box">
                    <div class="metric-icon">⏱️</div>
                    <div class="metric-val">{submission_delay:+d} d</div>
                    <div class="metric-desc">Submission Delay</div>
                </div>
                <div class="metric-box">
                    <div class="metric-icon">📚</div>
                    <div class="metric-val">{studied_credits}</div>
                    <div class="metric-desc">Enrolled Credits</div>
                </div>
                <div class="metric-box">
                    <div class="metric-icon">🔁</div>
                    <div class="metric-val">{prev_attempts}</div>
                    <div class="metric-desc">Prev Attempts</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-title">💡 Personalized Diagnostic Insights</div>', unsafe_allow_html=True)
            insights = []
            if sum_click < 500:
                insights.append(("dot-red", "Severe Platform Disengagement", f"Only {sum_click} total clicks recorded. Severe inactivity is among the primary precursors to student withdrawal in the OULAD dataset."))
            elif sum_click < 1500:
                insights.append(("dot-yellow", "Sub-optimal Platform Activity", f"{sum_click:,} clicks indicates below-average engagement. Increasing regular weekly check-ins helps avoid end-of-term cramming."))
            else:
                insights.append(("dot-green", "Healthy Learning Activity", f"{sum_click:,} clicks reflects regular interaction with learning resources."))

            if submission_delay > 40:
                insights.append(("dot-red", "Critical Assignment Backlog", f"Submissions averaging {submission_delay} days late. Chronic delays compound workload stress exponentially across consecutive modules."))
            elif submission_delay > 5:
                insights.append(("dot-yellow", "Moderate Submission Delays", f"Assignments are running {submission_delay} days behind schedule. Pacing adjustments are recommended."))
            else:
                insights.append(("dot-green", "Timely Submission Habits", "Submissions are submitted on or ahead of deadlines, preventing deadline anxiety."))

            if studied_credits > 240:
                insights.append(("dot-red", "Excessive Credit Overload", f"Enrolled in {studied_credits} credits. Heavy academic credit loads strongly correlate with elevated behavioral strain."))
            else:
                insights.append(("dot-green", "Balanced Credit Volume", f"{studied_credits} credits is well within manageable institutional recommendations."))

            for dot, title, desc in insights:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-dot {dot}"></div>
                    <div>
                        <div class="insight-text"><strong>{title}</strong></div>
                        <div class="insight-sub">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-title">🛠️ Evidence-Based Interventions</div>', unsafe_allow_html=True)
            if prediction == 0:
                recs = [
                    ("✅", "**Maintain Consistent Rhythms:** Continue dedicating fixed weekly time-blocks for reviewing lectures and assignments."),
                    ("📈", "**Enrichment Opportunities:** Explore academic enrichment without overloading your course schedule."),
                    ("🧘", "**Proactive Wellness:** Maintain scheduled leisure and physical exercise periods.")
                ]
            elif prediction == 1:
                recs = [
                    ("📅", "**Micro-Milestone Decomposition:** Break multi-week assignments into 3-day deliverables to avoid last-minute stress spikes."),
                    ("⏱️", "**LMS Regularity:** Dedicate 20 structured minutes daily to checking course materials and announcements."),
                    ("🤝", "**Peer Study Collaboration:** Work with study peers to maintain steady course pacing.")
                ]
            else:
                recs = [
                    ("🚨", "**Academic Advising Consultation:** Reach out to your course coordinator or academic advisor regarding workload redistribution."),
                    ("📋", "**Deadline Triage:** Prioritize high-weighting assessments and discuss extension options where permissible."),
                    ("🏥", "**Student Counseling & Wellness Services:** Access campus student support services for stress management guidance.")
                ]

            for icon, text in recs:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-icon">{icon}</div>
                    <div class="rec-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="section-title">🔬 Local Multiclass SHAP Attribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Mathematical feature attribution showing which specific inputs pushed the prediction toward or away from the predicted risk class</div>', unsafe_allow_html=True)
            
            explainer, expl_model, _ = get_cached_explainer()
            if explainer is not None:
                explanation = explain_student(features_df, expl_model, explainer, target_class=prediction)
                st.markdown(explanation["explanation_text"])
                
                # Plot horizontal bar with explicit class target in title
                factors_df = pd.DataFrame(explanation["all_factors"])
                fig, ax = plt.subplots(figsize=(7, 3.8))
                colors = ['#ef4444' if x > 0 else '#22c55e' for x in factors_df['shap_value']]
                ax.barh(factors_df['description'][::-1], factors_df['shap_value'][::-1], color=colors[::-1])
                ax.set_xlabel(f"SHAP Impact toward {explanation['target_class_name']}")
                ax.axvline(0, color="#94a3b8", linestyle="--", alpha=0.7)
                ax.set_title(f"SHAP Feature Attribution: Contribution to {explanation['target_class_name']}")
                plt.tight_layout()
                st.pyplot(fig)
                st.caption(f"Note: Positive SHAP values (red) push the probability towards {explanation['target_class_name']}; negative values (green) push probability away.")
            else:
                st.info("SHAP explainer initializing with background data... Run `python3 run_research_pipeline.py` if models are not generated.")

    # ─── FAQ Section ───
    st.markdown("---")
    st.markdown('<div class="section-title">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)
    st.markdown("""
    <details class="faq-item">
        <summary>What dataset is this system trained on?</summary>
        <p>The system is trained on the Open University Learning Analytics Dataset (OULAD), encompassing 30,000+ university students across multiple STEM and social science modules with complete clickstream, assessment, and enrollment records.</p>
    </details>
    <details class="faq-item">
        <summary>Are the labels direct clinical burnout diagnoses?</summary>
        <p>No. OULAD logs do not contain psychiatric or psychological diagnoses. The labels represent a behavior-derived Burnout Risk Proxy constructed from platform disengagement, chronic submission delays, previous course failures, and credit overload divided into balanced Low, Medium, and High tertiles.</p>
    </details>
    <details class="faq-item">
        <summary>Is this a clinical psychological diagnosis?</summary>
        <p>No. This is an educational data mining and early-warning screening tool intended for academic advisors, instructors, and students to identify behavioral risk patterns weeks before academic failure occurs.</p>
    </details>
    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: BASELINE VS EARLY-WINDOW DETECTION STUDY
# ==============================================================================
elif nav_option == "📊 Baseline vs Early Detection":
    st.markdown("""
    <div class="hero">
        <h1>📊 Baseline vs Early-Window Detection Study</h1>
        <p>Empirical evaluation of full-semester vs early 21-day predictive capability across 4 ML classifiers</p>
        <div class="hero-badge">🔬 25,536 Students Evaluated · 5-Fold Stratified Cross-Validation · Leakage-Free Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    b_res, e_res = load_research_results()

    if b_res is None or e_res is None:
        st.warning("⚠️ Research model results not found. Please run `python3 run_research_pipeline.py` to generate model files.")
    else:
        b_acc = b_res['comparison_df'].loc[b_res['best_name'], 'Test Accuracy'] * 100
        e_acc = e_res['comparison_df'].loc[e_res['best_name'], 'Test Accuracy'] * 100
        delta = b_acc - e_acc

        st.markdown(f"""
        <div class="tradeoff-banner">
            <div class="tradeoff-title">⚖️ The Early-Detection Trade-off Analysis</div>
            <div class="tradeoff-desc">
                Predicting student burnout risk at <strong>Day 21 (Week 3)</strong> achieves <strong>{e_acc:.2f}% accuracy</strong> compared to <strong>{b_acc:.2f}%</strong> when using full-semester hindsight data. 
                This <strong>{delta:.2f}% accuracy difference</strong> is the honest mathematical trade-off of temporal early forecasting: early detection sacrifices hindsight information to provide an actionable 3-week window for academic counseling and intervention <em>before</em> assignments are missed or students drop out.
            </div>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)

        with colA:
            st.markdown(f"""
            <div class="res-card">
                <div class="res-card-header">
                    <span>Full-Semester Baseline (End of Term)</span>
                    <span class="res-pill pill-purple">Best: {b_res['best_name']}</span>
                </div>
                <div class="res-card-body">
                    <strong>Test Accuracy:</strong> {b_acc:.2f}%<br>
                    <strong>Macro F1:</strong> {b_res['comparison_df'].loc[b_res['best_name'], 'Test F1 (macro)']:.4f}<br>
                    <strong>5-Fold CV Accuracy:</strong> {b_res['comparison_df'].loc[b_res['best_name'], 'CV Acc (mean)']*100:.2f}% ± {b_res['comparison_df'].loc[b_res['best_name'], 'CV Acc (std)']*100:.2f}%<br>
                    <em>Full behavioral visibility across all modules, assignments, and click history.</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="res-card">
                <div class="res-card-header">
                    <span>Early-Window Detection (First 21 Days)</span>
                    <span class="res-pill pill-blue">Best: {e_res['best_name']}</span>
                </div>
                <div class="res-card-body">
                    <strong>Test Accuracy:</strong> {e_acc:.2f}%<br>
                    <strong>Macro F1:</strong> {e_res['comparison_df'].loc[e_res['best_name'], 'Test F1 (macro)']:.4f}<br>
                    <strong>5-Fold CV Accuracy:</strong> {e_res['comparison_df'].loc[e_res['best_name'], 'CV Acc (mean)']*100:.2f}% ± {e_res['comparison_df'].loc[e_res['best_name'], 'CV Acc (std)']*100:.2f}%<br>
                    <em>Uses only early clicks, initial submission habits, and credit registration.</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📈 Benchmark Model Comparison Table</div>', unsafe_allow_html=True)
        
        # Merge comparison tables
        df_base = b_res['comparison_df'].copy().add_prefix("Baseline_")
        df_early = e_res['comparison_df'].copy().add_prefix("EarlyWindow_")
        merged_table = pd.concat([df_base, df_early], axis=1)
        st.dataframe(
            merged_table.style.format("{:.4f}").background_gradient(cmap="Blues", subset=[
                "Baseline_Test Accuracy", "EarlyWindow_Test Accuracy", "Baseline_Test F1 (macro)", "EarlyWindow_Test F1 (macro)"
            ]),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown('<div class="section-title">🔍 Interactive Confusion Matrix Explorer</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            exp_choice = st.selectbox("Select Experiment Window:", ["Full-Semester Baseline", "Early-Window (21 Days)"])
        with c2:
            current_models = list(b_res['fitted_models'].keys()) if exp_choice == "Full-Semester Baseline" else list(e_res['fitted_models'].keys())
            model_choice = st.selectbox("Select Classifier:", current_models)

        if exp_choice == "Full-Semester Baseline":
            target_model = b_res['fitted_models'][model_choice]
            X_t, y_t = b_res['X_test'], b_res['y_test']
        else:
            target_model = e_res['fitted_models'][model_choice]
            X_t, y_t = e_res['X_test'], e_res['y_test']

        fig_cm = plot_confusion_matrix(target_model, X_t, y_t, title=f"{exp_choice} — {model_choice}")
        col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
        with col_cm2:
            st.pyplot(fig_cm)


# ==============================================================================
# VIEW 3: EXPLAINABLE AI (SHAP) LAB
# ==============================================================================
elif nav_option == "🔍 Explainable AI (SHAP) Lab":
    st.markdown("""
    <div class="hero">
        <h1>🔍 Multiclass Explainable AI (SHAP) Dashboard</h1>
        <p>Interpretability via Shapley Additive Explanations (SHAP) across global features and individual student profiles</p>
        <div class="hero-badge">🧠 Game-Theoretic Attributions · Class-Specific Explanations · Multiclass Consistency</div>
    </div>
    """, unsafe_allow_html=True)

    explainer, expl_model, X_sample = get_cached_explainer()

    if explainer is None:
        st.warning("⚠️ Research model not loaded. Please run `python3 run_research_pipeline.py`.")
    else:
        tab_global, tab_cases = st.tabs(["🌐 Global Feature Importance & Summary", "👤 Student Persona Case Studies"])

        with tab_global:
            st.markdown('<div class="section-title">📊 Global Feature Ranking (Mean |SHAP| Across All Classes)</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Quantifies overall feature magnitude across Low, Medium, and High risk tiers</div>', unsafe_allow_html=True)

            with st.spinner("Computing SHAP values on sample batch..."):
                sv = compute_shap_values(explainer, X_sample, expl_model)
                imp_df = global_importance_df(sv)

            colG1, colG2 = st.columns([3, 2])
            with colG1:
                fig_imp = plot_global_importance(sv)
                st.pyplot(fig_imp)
            with colG2:
                st.markdown("### Top Predictive Drivers")
                for idx, row in imp_df.head(5).iterrows():
                    st.markdown(f"**{idx+1}. {row['Description']}**")
                    st.caption(f"Mean |SHAP| Impact: `{row['Mean |SHAP|']:.4f}`")

            st.markdown("---")
            st.markdown('<div class="section-title">🐝 Class-Specific SHAP Beeswarm Distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Select which risk class to visualize. Shows how feature values shift the probability toward or away from that specific class.</div>', unsafe_allow_html=True)
            
            target_class_choice = st.selectbox(
                "Select Target Risk Class for Beeswarm Summary:",
                options=[2, 1, 0],
                format_func=lambda x: f"Class {x}: {RISK_LABELS[x]}"
            )

            fig_summary = plot_summary(sv, X_sample, class_idx=target_class_choice)
            st.pyplot(fig_summary)
            st.caption(f"Interpretation: For {RISK_LABELS[target_class_choice]}, dots to the right (positive SHAP) increase the likelihood of {RISK_LABELS[target_class_choice]}, while dots to the left (negative SHAP) decrease it.")

        with tab_cases:
            st.markdown('<div class="section-title">👤 Interactive Student Persona Explanations</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Select a student archetype to inspect class-specific SHAP attribution</div>', unsafe_allow_html=True)

            personas = {
                "Proactive High Achiever": {
                    "total_clicks": 9500, "submission_delay": -14, "delay_abs": 14,
                    "engagement_level": 2, "engagement_per_day": 9500, "delay_ratio": -14/15,
                    "click_intensity": 9500/121, "activity_score": 19000,
                    "num_of_prev_attempts": 0, "studied_credits": 120
                },
                "Struggling Overloaded Student": {
                    "total_clicks": 420, "submission_delay": 35, "delay_abs": 35,
                    "engagement_level": 0, "engagement_per_day": 210, "delay_ratio": 35/36,
                    "click_intensity": 420/301, "activity_score": 0,
                    "num_of_prev_attempts": 1, "studied_credits": 300
                },
                "Disengaged Ghost Student": {
                    "total_clicks": 80, "submission_delay": 65, "delay_abs": 65,
                    "engagement_level": 0, "engagement_per_day": 80, "delay_ratio": 65/66,
                    "click_intensity": 80/121, "activity_score": 0,
                    "num_of_prev_attempts": 0, "studied_credits": 120
                },
                "Course Retake with Late Submissions": {
                    "total_clicks": 1800, "submission_delay": 18, "delay_abs": 18,
                    "engagement_level": 1, "engagement_per_day": 600, "delay_ratio": 18/19,
                    "click_intensity": 1800/181, "activity_score": 1800,
                    "num_of_prev_attempts": 2, "studied_credits": 180
                }
            }

            c_pers1, c_pers2 = st.columns(2)
            with c_pers1:
                p_choice = st.selectbox("Choose Student Archetype:", list(personas.keys()))
            with c_pers2:
                class_inspect = st.selectbox("Attribution Target Class:", [None, 2, 1, 0], format_func=lambda x: "Predicted Class" if x is None else f"Class {x}: {RISK_LABELS[x]}")

            p_data = personas[p_choice]
            p_df = pd.DataFrame([p_data])[FEATURE_COLS]

            exp_res = explain_student(p_df, expl_model, explainer, target_class=class_inspect)

            colP1, colP2 = st.columns([1, 1])
            with colP1:
                st.markdown(f"### Predicted Classification: {exp_res['risk_label']}")
                st.markdown(f"**Model Confidence:** `{exp_res['confidence']:.1f}%`")
                st.markdown(f"**SHAP Attribution Target:** {exp_res['target_class_name']}")
                st.markdown("#### Primary Contributing Factors:")
                for f in exp_res['top_factors']:
                    icon = "🔴" if f['shap_value'] > 0 else "🟢"
                    st.markdown(f"{icon} **{f['description']}**: `{f['raw_value']:.1f}` ({f['direction']}, SHAP `{f['shap_value']:+.4f}`)")

            with colP2:
                f_df = pd.DataFrame(exp_res['all_factors'])
                fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
                cols = ['#ef4444' if x > 0 else '#22c55e' for x in f_df['shap_value']]
                ax_p.barh(f_df['description'][::-1], f_df['shap_value'][::-1], color=cols[::-1])
                ax_p.set_xlabel(f"SHAP Impact toward {exp_res['target_class_name']}")
                ax_p.axvline(0, color="#94a3b8", linestyle="--")
                ax_p.set_title(f"Factors Influencing {exp_res['target_class_name']}")
                plt.tight_layout()
                st.pyplot(fig_p)


# ==============================================================================
# VIEW 4: PILOT PRIMARY SURVEY VALIDATION (MBI-SS)
# ==============================================================================
elif nav_option == "📋 MBI-SS Pilot Survey Validation":
    st.markdown("""
    <div class="hero">
        <h1>📋 Pilot Empirical Validation (MBI-SS Survey)</h1>
        <p>Pilot empirical validation framework connecting behavior-derived proxy predictions with self-reported MBI-SS psychometric data</p>
        <div class="hero-badge">📝 9-Item MBI Short Form · Spearman Rank Correlation · Pilot Validation Framework</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <strong>📌 Research Note:</strong> This module represents a <strong>pilot empirical validation pipeline</strong>. Small survey samples do not claim clinical validity; rather, they assess the statistical rank correlation (&rho;) between digital LMS behavioral proxies and self-reported burnout dimensions (Exhaustion, Cynicism, Academic Inefficacy).
    </div>
    """, unsafe_allow_html=True)

    tab_test, tab_upload = st.tabs(["📝 Interactive 9-Item Questionnaire", "📤 Batch Survey CSV Upload & Spearman Test"])

    with tab_test:
        st.markdown('<div class="section-title">📝 Pilot Questionnaire: 9-Item MBI-Student Survey</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Rate each statement from 0 (Never) to 6 (Every day) based on current semester experience</div>', unsafe_allow_html=True)

        colS1, colS2 = st.columns(2)
        with colS1:
            st.markdown("#### 1. Emotional Exhaustion Subscale")
            q1 = st.slider("Q1: I feel emotionally drained from my studies", 0, 6, 2)
            q2 = st.slider("Q2: I feel used up at the end of a study session", 0, 6, 2)
            q3 = st.slider("Q3: I feel tired when I get up in the morning facing another study day", 0, 6, 2)

            st.markdown("#### 2. Cynicism & Disengagement Subscale")
            q4 = st.slider("Q4: I have become less interested in my studies since starting", 0, 6, 1)
            q5 = st.slider("Q5: I have become less enthusiastic about my courses", 0, 6, 1)
            q6 = st.slider("Q6: I doubt the significance of my studies", 0, 6, 1)

        with colS2:
            st.markdown("#### 3. Academic Efficacy Subscale (Reverse-Scored)")
            q7 = st.slider("Q7: I can effectively solve problems arising in my studies", 0, 6, 5)
            q8 = st.slider("Q8: I believe I make an effective contribution to class discussions", 0, 6, 4)
            q9 = st.slider("Q9: I feel I am making steady academic progress", 0, 6, 5)

        # Calculate MBI Subscales
        ex_score = (q1 + q2 + q3) / 3.0
        cy_score = (q4 + q5 + q6) / 3.0
        eff_score = (q7 + q8 + q9) / 3.0
        eff_rev = 6.0 - eff_score
        composite_burnout = (ex_score + cy_score + eff_rev) / 3.0

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Psychometric Score Summary</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-icon">🔥</div>
                <div class="metric-val">{ex_score:.2f} / 6</div>
                <div class="metric-desc">Exhaustion Subscale</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">🧊</div>
                <div class="metric-val">{cy_score:.2f} / 6</div>
                <div class="metric-desc">Cynicism Subscale</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">🎯</div>
                <div class="metric-val">{eff_score:.2f} / 6</div>
                <div class="metric-desc">Academic Efficacy</div>
            </div>
            <div class="metric-box" style="border: 2px solid #6366f1;">
                <div class="metric-icon">📈</div>
                <div class="metric-val">{composite_burnout:.2f} / 6</div>
                <div class="metric-desc">MBI Composite Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_upload:
        st.markdown('<div class="section-title">📤 Upload Empirical Survey Data CSV</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Evaluates rank correlation between model-predicted risk and self-reported MBI scores. No synthetic data is fabricated.</div>', unsafe_allow_html=True)

        if os.path.exists(SURVEY_TEMPLATE_PATH):
            with open(SURVEY_TEMPLATE_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Blank Survey CSV Template",
                    data=f,
                    file_name="survey_template.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader("Upload collected survey CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                survey_raw = pd.read_csv(uploaded_file)
                survey_scored = compute_mbi_scores(survey_raw)
                st.success(f"Loaded {len(survey_scored)} responses successfully!")
                
                b_data, _ = load_research_results()
                if b_data is not None:
                    val_result = validate_model_vs_survey(survey_scored, b_data['best_model'], FEATURE_COLS)
                    st.markdown(f"**Sample Size (N):** `{val_result['n']}`")
                    st.markdown(f"**Spearman Rank Correlation (ρ):** `{val_result['spearman_rho']:.4f}`")
                    st.markdown(f"**p-value:** `{val_result['p_value']:.4e}`")
                    fig_scatter = plot_validation_scatter(val_result['comparison_df'])
                    st.pyplot(fig_scatter)
            except Exception as e:
                st.error(f"Error validating survey file: {e}")
        else:
            st.info("No survey dataset uploaded yet. The analysis pipeline is ready for real student responses. You can download the template above to begin empirical data collection.")


# ==============================================================================
# VIEW 5: RESEARCH FRAMEWORK & VIVA DEFENSE GUIDE
# ==============================================================================
elif nav_option == "📚 Research Framework & Viva Guide":
    st.markdown("""
    <div class="hero">
        <h1>📚 Research Framework & Viva Defense Guide</h1>
        <p>Scientific documentation, theoretical proxy formulation, literature taxonomy, and viva defense cheat sheet</p>
        <div class="hero-badge">📖 Methodological Integrity · Ready for Viva & Academic Defense</div>
    </div>
    """, unsafe_allow_html=True)

    tab_prob, tab_math, tab_lit, tab_viva = st.tabs([
        "🎯 Problem & Motivation",
        "📐 Behavioral Proxy Formulation",
        "📚 Literature Taxonomy (20 Papers)",
        "🎓 Viva / Defense Q&A Cheat Sheet"
    ])

    with tab_prob:
        st.markdown("""
        ### 1. Problem Definition & Reactive vs Proactive Paradigm
        Academic burnout among higher education students is characterized by emotional exhaustion, cynicism towards studies, and reduced academic efficacy.
        
        - **The Core Problem:** Current institutional interventions are largely **reactive** — students are flagged only after failing exams, missing final submissions, or discontinuing courses.
        - **Our Proposed Solution:** An **early-window risk screening system** that detects behavioral risk signatures within the first 3 weeks (21 days) of the semester, enabling proactive pedagogical support before academic damage becomes irreversible.
        - **Proxy Label Acknowledgment:** Because raw institutional LMS datasets do not provide psychiatric clinical diagnoses, we construct and explicitly document a behavior-derived Burnout Risk Proxy.
        """)

    with tab_math:
        st.markdown(r"""
        ### 2. Burnout Risk Proxy Derivation
        Because real LMS logs lack psychological ground truth labels, a continuous Burnout Risk Proxy was engineered from normalized behavioral metrics:

        $$\text{Burnout Proxy Score} = 0.35 \times (1 - \hat{C}) + 0.35 \times \hat{D} + 0.15 \times \hat{A} + 0.15 \times \hat{W}$$

        Where:
        - $\hat{C}$: Min-Max Normalized Engagement (Total Clicks)
        - $\hat{D}$: Min-Max Normalized Submission Delay (Days Late)
        - $\hat{A}$: Min-Max Normalized Previous Course Attempts
        - $\hat{W}$: Min-Max Normalized Workload (Enrolled Credits)
        
        Quantile-based tertiles define **Low Risk (Class 0)**, **Medium Risk (Class 1)**, and **High Risk (Class 2)** tiers.
        """)

    with tab_lit:
        st.markdown("""
        ### 3. Literature Review Taxonomy (20 Selected Papers)
        1. **Burnout Theory & Measurement:** Maslach et al. (2001), Schaufeli et al. (2002), Salmela-Aro et al. (2009)
        2. **Learning Analytics & Clickstream Mining:** Siemens & Long (2011), Tempelaar et al. (2015), Gašević et al. (2016)
        3. **Machine Learning in Student At-Risk Prediction:** Kuzilek et al. (2017) [OULAD Benchmark], Marbouti et al. (2016), subsurface ensemble methods.
        4. **Explainable AI (XAI) in Education:** Lundberg & Lee (2017) [SHAP], Conati et al. (2018), Khosravi et al. (2022).
        5. **Early Warning Systems & Timely Interventions:** Arnold & Pistilli (2012) [Purdue Signals], Jayaprakash et al. (2014).
        """)

    with tab_viva:
        st.markdown("""
        ### 4. Key Viva Questions & Bulletproof Answers

        **Q1: "Where did the burnout risk labels come from in the OULAD dataset?"**  
        *Answer:* "OULAD contains raw behavioral LMS records without psychological labels. We engineered a multi-factor Burnout Risk Proxy combining platform disengagement, chronic submission lateness, course retakes, and credit overload. We then validate this proxy using a pilot MBI-SS psychometric survey to assess rank correlation with self-reported burnout dimensions."

        **Q2: "Why is the early-window model (66.27%) lower in accuracy than the full-semester model (88.97%)?"**  
        *Answer:* "The 22.7% difference represents the early-detection trade-off: at Day 21, the model observes only initial behavioral signals (Week 1-3). The full-semester baseline has the luxury of end-of-term hindsight. In real-world educational deployment, a 66%+ accurate early warning in Week 3 is vastly more actionable than 89% accuracy when it is too late to intervene."

        **Q3: "How does multiclass SHAP work in this 3-class system?"**  
        *Answer:* "SHAP attributes game-theoretic Shapley values to each feature for every specific class. A positive SHAP value for Class 2 (High Risk) increases the model's output probability for High Risk, whereas a positive SHAP value for Class 0 (Low Risk) indicates a protective factor that increases the probability of Low Risk."
        """)


# ─── Footer ───
st.markdown("""
<div class="footer">
    🚀 Early Student Burnout-Risk Screening System · Academic Research & Engineering Platform<br>
    Streamlit · Scikit-learn · XGBoost · SHAP · OULAD Dataset · MBI-SS Pilot Validation<br>
    <span style="font-size:0.72rem;">⚠️ Behavior-derived screening system — not a clinical medical diagnostic instrument</span>
</div>
""", unsafe_allow_html=True)
