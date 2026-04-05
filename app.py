import streamlit as st
import joblib
import numpy as np

# ─── Page Config ───
st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="🎓",
    layout="centered"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; max-width: 880px; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 100%);
        padding: 2.8rem 2rem 2.2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; top: -40%; right: -20%;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute; bottom: -30%; left: -10%;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; color: white; position: relative; z-index: 1; }
    .hero p  { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; position: relative; z-index: 1; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.78rem;
        margin-top: 0.8rem;
        backdrop-filter: blur(4px);
        position: relative; z-index: 1;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1e293b;
        margin: 1.8rem 0 0.8rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .section-subtitle {
        font-size: 0.85rem; color: #94a3b8; margin-top: -0.5rem; margin-bottom: 1rem;
    }

    /* ── Input cards ── */
    .input-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1.3rem 0.6rem;
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
        font-size: 0.76rem; color: #94a3b8; margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .input-example {
        font-size: 0.72rem; color: #a78bfa;
        background: #f5f3ff; padding: 0.3rem 0.6rem;
        border-radius: 6px; display: inline-block; margin-bottom: 0.4rem;
    }

    /* ── Result cards ── */
    .result-card {
        padding: 2.2rem 1.5rem; border-radius: 18px; text-align: center;
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
        background: linear-gradient(145deg, #fecaca 0%, #fca5a5 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 6px 20px rgba(239,68,68,0.2);
    }
    .result-emoji { font-size: 3.5rem; margin-bottom: 0.3rem; }
    .result-level { font-size: 1.6rem; font-weight: 800; color: #1e293b; }
    .result-advice { font-size: 0.92rem; color: #475569; margin-top: 0.4rem; line-height: 1.5; }
    .result-conf {
        font-size: 0.8rem; color: #64748b; margin-top: 0.6rem;
        background: rgba(255,255,255,0.5); display: inline-block;
        padding: 0.25rem 0.8rem; border-radius: 10px;
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
    .metric-val { font-size: 1.4rem; font-weight: 800; color: #1e293b; }
    .metric-desc { font-size: 0.7rem; color: #94a3b8; margin-top: 0.15rem; font-weight: 500; }
    .metric-icon { font-size: 1.3rem; margin-bottom: 0.2rem; }

    /* ── Insight items ── */
    .insight-card {
        display: flex; align-items: flex-start; gap: 0.8rem;
        padding: 0.85rem 1rem; margin: 0.4rem 0;
        border-radius: 10px; background: #ffffff;
        border: 1px solid #f1f5f9;
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
    .insight-sub  { font-size: 0.75rem; color: #94a3b8; margin-top: 0.15rem; }

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

    /* ── How it works ── */
    .how-step {
        display: flex; align-items: flex-start; gap: 1rem;
        padding: 0.8rem 0;
    }
    .step-num {
        width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; font-weight: 700; font-size: 0.85rem;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .step-content { flex: 1; }
    .step-title { font-weight: 600; font-size: 0.9rem; color: #1e293b; }
    .step-desc  { font-size: 0.8rem; color: #64748b; margin-top: 0.1rem; }

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
        font-size: 0.75rem; color: #cbd5e1;
        border-top: 1px solid #f1f5f9; margin-top: 2.5rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #faf5ff 0%, #f5f3ff 100%);
    }

    /* ── Tabs styling ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 🎓 About")
    st.markdown(
        "This AI system predicts how likely a student is "
        "to experience **burnout** — a state of emotional, mental, and physical exhaustion "
        "caused by academic overload."
    )

    st.markdown("---")

    st.markdown("### 🧠 How It Works")
    st.markdown("""
    <div class="how-step">
        <div class="step-num">1</div>
        <div class="step-content">
            <div class="step-title">Enter Your Info</div>
            <div class="step-desc">Move the sliders to match your academic situation</div>
        </div>
    </div>
    <div class="how-step">
        <div class="step-num">2</div>
        <div class="step-content">
            <div class="step-title">AI Analyzes Patterns</div>
            <div class="step-desc">The model compares your data against thousands of students</div>
        </div>
    </div>
    <div class="how-step">
        <div class="step-num">3</div>
        <div class="step-content">
            <div class="step-title">Get Your Results</div>
            <div class="step-desc">See your risk level, insights, and personalized tips</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Model Details")
    st.markdown(
        "**Algorithm:** Random Forest Classifier  \n"
        "**Dataset:** OULAD (Open University)  \n"
        "**Features:** 10 engineered metrics  \n"
        "**Accuracy:** ~88%  \n"
        "**Classes:** Low · Medium · High"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#a0aec0; text-align:center;'>"
        "Built as an academic ML project<br>"
        "Not a medical diagnostic tool</div>",
        unsafe_allow_html=True
    )


# ─── Hero ───
st.markdown("""
<div class="hero">
    <h1>🎓 Student Burnout Predictor</h1>
    <p>Understand your academic stress level using AI — in just 30 seconds</p>
    <div class="hero-badge">✨ Powered by Machine Learning · Trained on 30,000+ student records</div>
</div>
""", unsafe_allow_html=True)


# ─── Load Model ───
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception:
    st.error("⚠️ Could not load `model.pkl`. Make sure it's in the same directory as `app.py`.")
    st.stop()


# ─── Quick Guide (collapsible) ───
with st.expander("👋 First time here? Click to see a quick guide", expanded=False):
    st.markdown("""
    **What is this tool?**  
    This tool predicts whether you're at risk of academic burnout based on 4 simple inputs about your study habits.

    **Who is it for?**  
    Students, teachers, academic counselors — anyone who wants to check academic stress levels.

    **Is it accurate?**  
    The AI model was trained on real university data and achieves ~88% accuracy. It's a screening tool, not a medical diagnosis.

    **What do I need to do?**  
    Just move the 4 sliders below to match your situation and hit **Predict**. That's it!
    """)


# ─── Input Section ───
st.markdown('<div class="section-title">📝 Enter Your Academic Details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Adjust the sliders to match your current academic situation</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="input-card">
        <div class="input-label">🖱️ Engagement (Clicks)</div>
        <div class="input-helper">How many times you've interacted with the online learning platform (VLE) — clicking lectures, quizzes, materials, etc.</div>
        <div class="input-example">💡 Example: 500 = rarely active · 5000 = very active · 10000+ = power user</div>
    </div>
    """, unsafe_allow_html=True)
    sum_click = st.slider("Engagement Clicks", 0, 15000, 3000, step=100, label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class="input-card">
        <div class="input-label">⏱️ Submission Delay</div>
        <div class="input-helper">On average, how many days before or after the deadline do you submit assignments? Negative = early, Positive = late.</div>
        <div class="input-example">💡 Example: −15 = 15 days early · 0 = on time · +30 = 30 days late</div>
    </div>
    """, unsafe_allow_html=True)
    submission_delay = st.slider("Submission Delay", -200, 200, 0, step=1, label_visibility="collapsed")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="input-card">
        <div class="input-label">🔁 Previous Attempts</div>
        <div class="input-helper">How many times you've previously attempted this course/module before the current enrollment.</div>
        <div class="input-example">💡 Example: 0 = first time · 1 = retaking once · 3+ = multiple retakes</div>
    </div>
    """, unsafe_allow_html=True)
    prev_attempts = st.slider("Previous Attempts", 0, 6, 0, step=1, label_visibility="collapsed")

with col4:
    st.markdown("""
    <div class="input-card">
        <div class="input-label">📚 Studied Credits</div>
        <div class="input-helper">Total credit hours you're enrolled in this semester. Higher credits = heavier workload.</div>
        <div class="input-example">💡 Example: 60 = light load · 120 = normal · 300+ = very heavy</div>
    </div>
    """, unsafe_allow_html=True)
    studied_credits = st.slider("Studied Credits", 30, 600, 120, step=10, label_visibility="collapsed")


# ─── Derived Features ───
delay_abs = abs(submission_delay)
engagement_level = 0 if sum_click <= 1000 else (1 if sum_click <= 3000 else 2)
engagement_per_day = sum_click / (prev_attempts + 1)
delay_ratio = submission_delay / (delay_abs + 1)
click_intensity = sum_click / (studied_credits + 1)
activity_score = sum_click * engagement_level

# ─── Predict ───
st.markdown("")
predict_btn = st.button("🔍  Predict My Burnout Risk", use_container_width=True, type="primary")

if predict_btn:
    features = np.array([[
        sum_click, submission_delay, delay_abs, engagement_level,
        engagement_per_day, delay_ratio, click_intensity, activity_score,
        prev_attempts, studied_credits
    ]])

    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100

    # ── Label config ──
    config = {
        0: {
            "label": "Low Burnout Risk",
            "emoji": "😊",
            "css": "result-low",
            "advice": "Great news! Your academic patterns look healthy. You're managing your workload well with good engagement and timely submissions.",
            "gauge_pos": "13%",
            "gauge_color": "#22c55e",
        },
        1: {
            "label": "Medium Burnout Risk",
            "emoji": "😐",
            "css": "result-medium",
            "advice": "Caution — some stress signals detected. You may be stretching yourself thin. Consider adjusting your habits before things escalate.",
            "gauge_pos": "48%",
            "gauge_color": "#eab308",
        },
        2: {
            "label": "High Burnout Risk",
            "emoji": "😰",
            "css": "result-high",
            "advice": "Alert! Multiple burnout indicators are present. Your current academic load and behavior patterns suggest significant stress. Please reach out for support.",
            "gauge_pos": "83%",
            "gauge_color": "#ef4444",
        },
    }

    c = config[prediction]

    # ── Tabs for organized output ──
    tab1, tab2, tab3 = st.tabs(["📊 Result", "💡 Insights", "🛠️ Action Plan"])

    with tab1:
        # Result card
        conf_html = f'<div class="result-conf">Model confidence: {confidence:.1f}%</div>' if confidence else ""
        st.markdown(f"""
        <div class="result-card {c['css']}">
            <div class="result-emoji">{c['emoji']}</div>
            <div class="result-level">{c['label']}</div>
            <div class="result-advice">{c['advice']}</div>
            {conf_html}
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-bar">
                <div class="gauge-marker" style="left: {c['gauge_pos']}; background: {c['gauge_color']};"></div>
            </div>
            <div class="gauge-labels">
                <span>Low Risk</span><span>Medium Risk</span><span>High Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-icon">🖱️</div>
                <div class="metric-val">{sum_click:,}</div>
                <div class="metric-desc">Total Clicks</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">⏱️</div>
                <div class="metric-val">{submission_delay:+d}</div>
                <div class="metric-desc">Delay (days)</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">📚</div>
                <div class="metric-val">{studied_credits}</div>
                <div class="metric-desc">Credits</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">🔁</div>
                <div class="metric-val">{prev_attempts}</div>
                <div class="metric-desc">Prev. Attempts</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-title">💡 Personalized Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Here\'s what your numbers tell us about your academic health</div>', unsafe_allow_html=True)

        insights = []

        # Engagement
        if sum_click < 500:
            insights.append(("dot-red",
                "Very low platform engagement",
                f"Only {sum_click} clicks means you're barely using the learning platform. Students with this level of engagement are at significantly higher risk of falling behind."))
        elif sum_click < 1500:
            insights.append(("dot-yellow",
                "Below-average engagement",
                f"With {sum_click:,} clicks, you're using the platform less than most students. Aim for at least 2,000–3,000 interactions for optimal learning."))
        elif sum_click < 5000:
            insights.append(("dot-green",
                "Good engagement level",
                f"{sum_click:,} clicks shows you're actively using course materials. This is a positive indicator of academic involvement."))
        else:
            insights.append(("dot-green",
                "Excellent engagement",
                f"With {sum_click:,} clicks, you're in the top tier of platform users. This level of activity strongly correlates with academic success."))

        # Submission delay
        if submission_delay > 50:
            insights.append(("dot-red",
                "Severely late submissions",
                f"Averaging {submission_delay} days late is a major warning sign. Late submissions create a snowball effect — each one adds pressure to the next deadline."))
        elif submission_delay > 10:
            insights.append(("dot-yellow",
                "Submissions tend to be late",
                f"You're averaging {submission_delay} days past deadline. While occasional delays happen, a pattern of lateness can indicate building stress."))
        elif submission_delay > -5:
            insights.append(("dot-green",
                "On-time submissions",
                "You're submitting close to or right on deadline — this shows good time management and discipline."))
        else:
            insights.append(("dot-green",
                "Early submissions",
                f"Submitting {abs(submission_delay)} days early on average is excellent. This gives you buffer time and reduces last-minute stress."))

        # Credits
        if studied_credits > 300:
            insights.append(("dot-red",
                "Extremely heavy course load",
                f"{studied_credits} credits is well above the recommended maximum. This level of workload is one of the strongest predictors of burnout."))
        elif studied_credits > 180:
            insights.append(("dot-yellow",
                "Above-average workload",
                f"With {studied_credits} credits, you're carrying more than the typical student. Keep a close eye on your energy levels."))
        else:
            insights.append(("dot-green",
                "Manageable workload",
                f"{studied_credits} credits is within a healthy range. This gives you room to balance academics with rest and personal life."))

        # Previous attempts
        if prev_attempts >= 3:
            insights.append(("dot-red",
                "Multiple course retakes",
                f"{prev_attempts} previous attempts suggest ongoing academic challenges. Repeated retakes can be emotionally draining and increase burnout risk."))
        elif prev_attempts >= 1:
            insights.append(("dot-yellow",
                "Some course retakes",
                f"{prev_attempts} previous attempt(s) is common and not alarming, but these courses may need extra attention and study strategies."))
        else:
            insights.append(("dot-green",
                "First attempt — fresh start",
                "You haven't needed to retake any courses. This is the ideal starting position."))

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
        st.markdown('<div class="section-title">🛠️ Your Personalized Action Plan</div>', unsafe_allow_html=True)

        if prediction == 0:
            st.markdown('<div class="section-subtitle">You\'re doing well! Here are tips to stay on track.</div>', unsafe_allow_html=True)
            recs = [
                ("✅", "**Keep it up!** Your current habits are working. Maintain your engagement and submission discipline."),
                ("📅", "**Set weekly reviews** — spend 15 minutes every Sunday planning the week ahead to stay proactive."),
                ("🧘", "**Prevent burnout proactively** — even when things are going well, schedule regular breaks and social time."),
                ("📈", "**Challenge yourself** — since you have bandwidth, consider exploring research opportunities or extra projects."),
            ]
        elif prediction == 1:
            st.markdown('<div class="section-subtitle">Some adjustments can bring you back to a healthy zone.</div>', unsafe_allow_html=True)
            recs = []
            if submission_delay > 10:
                recs.append(("📅", "**Break assignments into chunks** — instead of one big deadline, set 3-4 mini-deadlines for each task. Start with just outlining the assignment on day 1."))
            if sum_click < 1500:
                recs.append(("📖", "**Set a daily study habit** — commit to just 20 minutes on the VLE platform each day. Small, consistent engagement beats sporadic cramming."))
            if studied_credits > 180:
                recs.append(("⚖️", "**Evaluate your course load** — talk to your academic advisor about whether you can redistribute credits across semesters."))
            if prev_attempts >= 1:
                recs.append(("🤝", "**Get support for retake courses** — join study groups or visit tutoring centers. A fresh approach often helps more than just repeating the same methods."))
            recs.append(("💬", "**Talk to someone** — share your workload concerns with a friend, mentor, or counselor. External perspective can help you prioritize."))
        else:
            st.markdown('<div class="section-subtitle">Urgent steps to reduce your burnout risk.</div>', unsafe_allow_html=True)
            recs = [
                ("🚨", "**Acknowledge the situation** — recognizing burnout is the first step. You're not lazy or weak — your workload may genuinely be unsustainable."),
            ]
            if submission_delay > 30:
                recs.append(("📋", "**Triage your deadlines** — list all upcoming deadlines, identify which ones are most critical, and focus only on those. It's okay to ask for extensions on others."))
            if sum_click < 1000:
                recs.append(("🖥️", "**Reconnect with course materials** — start with just one module per day. Watch a single lecture or read one chapter. Rebuild the habit gradually."))
            if studied_credits > 250:
                recs.append(("📉", "**Seriously consider dropping a course** — your credit load is very high. Dropping one course now is better than failing multiple courses later."))
            recs.append(("🧘", "**Practice the 20-20-20 rule** — every 20 minutes, take a 20-second break and look at something 20 feet away. Small breaks prevent mental fatigue."))
            recs.append(("🏥", "**Reach out for professional support** — visit your university's student wellness center or counseling services. They're trained for exactly this."))

        for icon, text in recs:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-icon">{icon}</div>
                <div class="rec-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── FAQ Section ───
st.markdown("---")
st.markdown('<div class="section-title">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)

st.markdown("""
<details class="faq-item">
    <summary>What does "burnout" mean in this context?</summary>
    <p>Academic burnout is a state of chronic stress that leads to physical and emotional exhaustion, cynicism about studies, and feelings of ineffectiveness. It's caused by prolonged periods of high workload without adequate rest.</p>
</details>

<details class="faq-item">
    <summary>How accurate is this prediction?</summary>
    <p>The model achieves approximately 88% accuracy on test data from real university students. However, it's a screening tool — not a clinical diagnosis. If you're experiencing severe stress, please consult a professional.</p>
</details>

<details class="faq-item">
    <summary>What data was used to train this model?</summary>
    <p>The model was trained on the OULAD (Open University Learning Analytics Dataset), which contains anonymized data from 30,000+ student interactions including VLE clicks, assignment submissions, and course information.</p>
</details>

<details class="faq-item">
    <summary>Can I use this for someone else?</summary>
    <p>Yes — teachers and counselors can use this tool to assess student risk by entering the student's academic data. It's designed to help identify students who may need additional support.</p>
</details>

<details class="faq-item">
    <summary>What should I do if I get "High Burnout Risk"?</summary>
    <p>Don't panic. This is a data-based estimate. Start by reviewing the personalized action plan, consider talking to your academic advisor about your workload, and reach out to your university's student support services if needed.</p>
</details>
""", unsafe_allow_html=True)


# ─── Footer ───
st.markdown("""
<div class="footer">
    🚀 Student Burnout Prediction System · AI/ML Academic Project<br>
    Built with Streamlit · Scikit-learn · XGBoost · OULAD Dataset<br>
    <span style="font-size:0.7rem;">⚠️ This is a research project, not a medical diagnostic tool</span>
</div>
""", unsafe_allow_html=True)
