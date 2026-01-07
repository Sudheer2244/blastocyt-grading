import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import base64
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Blastocyst Grading System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

IMG_SIZE = 224

#------------Functions------------
def get_inference(icm, te, exp):
    notes = []
    if icm >= 4:
        notes.append("🟢 **ICM:** Good inner cell mass – likely healthy embryoblast.")
    elif icm == 3:
        notes.append("🟡 **ICM:** Moderate ICM – acceptable but not ideal.")
    else:
        notes.append("🔴 **ICM:** Poor ICM – lower likelihood of successful implantation.")

    if te >= 4:
        notes.append("🟢 **TE:** Strong trophectoderm – better implantation probability.")
    elif te == 3:
        notes.append("🟡 **TE:** Average TE – may still be viable.")
    else:
        notes.append("🔴 **TE:** Weak TE – may reduce implantation chances.")

    if exp >= 4:
        notes.append("🟢 **Expansion:** Well-expanded blastocyst – good development.")
    elif exp == 3:
        notes.append("🟡 **Expansion:** Moderate expansion – monitor carefully.")
    else:
        notes.append("🔴 **Expansion:** Poor expansion – embryo may be underdeveloped.")

    if icm >= 4 and te >= 4 and exp >= 4:
        summary = "✅ **Overall:** High-quality blastocyst – strong transfer candidate."
        quality_level = "Excellent"
    elif icm >= 3 and te >= 3 and exp >= 3:
        summary = "⚠️ **Overall:** Medium-quality embryo – possible, but requires clinical judgement."
        quality_level = "Moderate"
    else:
        summary = "❌ **Overall:** Low-quality blastocyst – poor prognosis."
        quality_level = "Poor"

    return summary, notes, quality_level


def get_clinical_recommendations(icm, te, exp):
    recommendations = []
    avg_score = (icm + te + exp) / 3
    
    recommendations.append("### 🎯 Transfer Recommendations")
    if avg_score >= 4:
        recommendations.append("- **Priority Candidate:** This embryo shows excellent quality indicators")
        recommendations.append("- **Transfer Strategy:** Suitable for single embryo transfer (SET)")
        recommendations.append("- **Success Rate:** High probability of successful implantation (60-70%)")
        recommendations.append("- **Timing:** Optimal for fresh transfer or vitrification")
    elif avg_score >= 3:
        recommendations.append("- **Viable Candidate:** This embryo shows moderate quality")
        recommendations.append("- **Transfer Strategy:** Consider patient age and history")
        recommendations.append("- **Success Rate:** Moderate probability of implantation (40-50%)")
        recommendations.append("- **Timing:** May benefit from extended culture monitoring")
    else:
        recommendations.append("- **Limited Viability:** This embryo shows lower quality indicators")
        recommendations.append("- **Transfer Strategy:** Discuss alternatives with patient")
        recommendations.append("- **Success Rate:** Lower probability of implantation (<30%)")
        recommendations.append("- **Timing:** Consider extended culture to Day 6")
    
    recommendations.append("\n### 🔬 Parameter-Specific Guidance")
    
    if icm < 3:
        recommendations.append("**ICM Concerns:**")
        recommendations.append("- Review culture media composition and oxygen levels")
        recommendations.append("- Consider growth factor supplementation")
        recommendations.append("- Evaluate oocyte quality in future cycles")
    
    if te < 3:
        recommendations.append("**TE Concerns:**")
        recommendations.append("- Potential implantation challenges expected")
        recommendations.append("- Consider assisted hatching if transfer proceeds")
        recommendations.append("- Optimize luteal phase progesterone support")
    
    if exp < 3:
        recommendations.append("**Expansion Concerns:**")
        recommendations.append("- Embryo may benefit from additional 12-24 hours culture")
        recommendations.append("- Assess zona pellucida thickness")
    
    recommendations.append("\n### 📋 Follow-Up Actions")
    recommendations.append("- Schedule detailed consultation with reproductive endocrinologist")
    recommendations.append("- Document findings in comprehensive medical record")
    recommendations.append("- Discuss realistic expectations with patient")
    recommendations.append("- Consider PGT-A testing for future embryos if indicated")
    
    return recommendations


def calculate_success_metrics(icm, te, exp):
    avg_score = (icm + te + exp) / 3
    base_prob = (icm * 0.35 + te * 0.40 + exp * 0.25) / 5 * 100
    
    if icm >= 4 and te >= 4:
        base_prob += 10
    if exp == 5:
        base_prob += 5
    
    base_prob = min(base_prob, 95)
    
    if exp >= 4:
        stage = "Advanced Development"
    elif exp == 3:
        stage = "Standard Development"
    else:
        stage = "Early Development"
    
    return {
        'success_probability': f"{base_prob:.1f}%",
        'quality_score': f"{avg_score:.2f}/5.00",
        'development_stage': stage,
        'transfer_priority': 'High' if avg_score >= 4 else 'Medium' if avg_score >= 3 else 'Low'
    }


def generate_text_report(icm, te, exp, patient_data):
    report_lines = []
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    report_lines.append("=" * 80)
    report_lines.append("BLASTOCYST GRADING REPORT".center(80))
    report_lines.append("AI-Powered Embryo Quality Assessment".center(80))
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report Generated: {timestamp}")
    report_lines.append(f"Report ID: {datetime.now().strftime('%Y%m%d-%H%M%S')}")
    report_lines.append("System Version: v2.0")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("PATIENT & EMBRYO INFORMATION")
    report_lines.append("-" * 80)
    
    for key, value in patient_data.items():
        if value and value != "N/A":
            report_lines.append(f"{key}: {value}")
    
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("GRADING RESULTS")
    report_lines.append("-" * 80)
    
    metrics = calculate_success_metrics(icm, te, exp)
    
    report_lines.append(f"Inner Cell Mass (ICM): {icm}/5 {'⭐' * icm}{'☆' * (5-icm)}")
    report_lines.append(f"Trophectoderm (TE): {te}/5 {'⭐' * te}{'☆' * (5-te)}")
    report_lines.append(f"Expansion Grade (EXP): {exp}/5 {'⭐' * exp}{'☆' * (5-exp)}")
    report_lines.append("")
    report_lines.append(f"Quality Score: {metrics['quality_score']}")
    report_lines.append(f"Success Probability: {metrics['success_probability']}")
    report_lines.append(f"Development Stage: {metrics['development_stage']}")
    report_lines.append(f"Transfer Priority: {metrics['transfer_priority']}")
    report_lines.append("")
    
    report_lines.append("-" * 80)
    report_lines.append("CLINICAL ANALYSIS")
    report_lines.append("-" * 80)
    
    summary, notes, quality_level = get_inference(icm, te, exp)
    clean_summary = summary.replace('**', '').replace('✅', '').replace('⚠️', '').replace('❌', '')
    report_lines.append(clean_summary)
    report_lines.append("")
    
    for note in notes:
        clean_note = note.replace('**', '').replace('🟢', '[GOOD]').replace('🟡', '[MODERATE]').replace('🔴', '[POOR]')
        report_lines.append(clean_note)
    
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("CLINICAL RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    recommendations = get_clinical_recommendations(icm, te, exp)
    for rec in recommendations:
        clean_rec = rec.replace('###', '').replace('**', '').replace('🎯', '').replace('🔬', '').replace('📋', '')
        report_lines.append(clean_rec)
    
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("GRADING CRITERIA REFERENCE")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("Inner Cell Mass (ICM): Grade 5 (Excellent) to Grade 1 (Poor)")
    report_lines.append("Trophectoderm (TE): Grade 5 (Excellent) to Grade 1 (Poor)")
    report_lines.append("Expansion (EXP): Grade 5 (Hatching) to Grade 1 (Early)")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DISCLAIMER: AI-assisted analysis. Requires professional validation.")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def save_to_history(analysis_data):
    st.session_state.history.insert(0, analysis_data)
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[:100]


# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.98);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
        color: #000000;
    }
    
    .info-card h3 {
        color: #1F3C88 !important;
        font-weight: 700;
    }
    
    .recommendation-box {
        background: rgba(255, 255, 255, 0.98);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #FFA500;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        color: #000000;
    }
    
    .recommendation-box h4 {
        color: #1F3C88 !important;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    .recommendation-box p {
        color: #2c3e50 !important;
        line-height: 1.8;
        margin: 8px 0;
        font-size: 15px;
    }
    
    .analysis-note {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #000000 !important;
        font-size: 16px;
        font-weight: 500;
    }
    
    .success-prob {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)


# Background
def add_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{encoded}");
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

add_bg("background 2.jpg")


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("embryo_output_model.keras")

model = load_model()


# Prediction function
def predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(img / 255.0, axis=0)
    icm, te, exp = model.predict(x, verbose=0)
    return (
        int(np.argmax(icm[0]) + 1),
        int(np.argmax(te[0]) + 1),
        int(np.argmax(exp[0]) + 1)
    )


# Sidebar
with st.sidebar:
    st.markdown("## 🧬 Navigation")
    page = st.radio("", ["🔬 Analysis", "📜 History", "📊 Statistics", "ℹ️ About"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_recommendations = st.checkbox("Show Recommendations", value=True)
    show_metrics = st.checkbox("Show Success Metrics", value=True)
    auto_save = st.checkbox("Auto-save History", value=True)
    
    st.markdown("---")
    if st.session_state.history:
        st.metric("Total Analyses", len(st.session_state.history))
        avg_quality = np.mean([h['avg_score'] for h in st.session_state.history])
        st.metric("Avg Quality", f"{avg_quality:.2f}/5")


# Main Content
if page == "🔬 Analysis":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1 style='color:#1F3C88; text-shadow:0px 2px 6px rgba(0,0,0,0.25);'>
                🧬 Blastocyst Grading System
            </h1>
            <p style='color:#666; font-size: 18px;'>AI-Powered Embryo Quality Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Patient/Embryo Details
    with st.expander("📋 Embryo & Patient Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="PT-2024-001")
            patient_age = st.number_input("Age", 18, 50, 32)
            clinic_name = st.text_input("Clinic", placeholder="Optional")
        
        with col2:
            embryo_id = st.text_input("Embryo ID", placeholder="EMB-001")
            cycle_day = st.selectbox("Development Day", ["Day 3", "Day 5", "Day 6"], index=1)
            collection_date = st.date_input("Collection Date")
        
        with col3:
            doctor_name = st.text_input("Embryologist", placeholder="Name")
            culture_media = st.text_input("Culture Media", placeholder="Optional")
            notes = st.text_area("Notes", placeholder="Observations...", height=100)
    
    # File Upload
    st.markdown("### 📤 Upload Embryo Image")
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded Embryo Image", use_container_width=True)
        
        if st.button("🔬 Analyze Embryo", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                icm, te, exp = predict(img)
                
                patient_data = {
                    "Patient ID": patient_id or "N/A",
                    "Age": patient_age,
                    "Embryo ID": embryo_id or "N/A",
                    "Day": cycle_day,
                    "Collection Date": collection_date.strftime("%Y-%m-%d"),
                    "Embryologist": doctor_name or "N/A",
                    "Clinic": clinic_name or "N/A",
                    "Media": culture_media or "N/A",
                    "Notes": notes or "None"
                }
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                avg_score = (icm + te + exp) / 3
                
                if auto_save:
                    save_to_history({
                        'timestamp': timestamp,
                        'patient_data': patient_data,
                        'icm': icm,
                        'te': te,
                        'exp': exp,
                        'avg_score': avg_score
                    })
                
                st.success("✅ Analysis Complete!")
                
                # Results
                st.markdown("## 📊 Grading Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>ICM</h4>
                            <h1>{icm}/5</h1>
                            <p>{'⭐' * icm}{'☆' * (5-icm)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>TE</h4>
                            <h1>{te}/5</h1>
                            <p>{'⭐' * te}{'☆' * (5-te)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>EXP</h4>
                            <h1>{exp}/5</h1>
                            <p>{'⭐' * exp}{'☆' * (5-exp)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Average</h4>
                            <h1>{avg_score:.2f}</h1>
                            <p>Quality</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Success Metrics
                if show_metrics:
                    metrics = calculate_success_metrics(icm, te, exp)
                    
                    st.markdown(f"""
                        <div class='success-prob'>
                            🎯 Success Probability: {metrics['success_probability']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Stage", metrics['development_stage'])
                    col2.metric("Priority", metrics['transfer_priority'])
                    col3.metric("Score", metrics['quality_score'])
                
                # Clinical Analysis
                st.markdown("## 🔬 Clinical Analysis")
                summary, notes_list, quality_level = get_inference(icm, te, exp)
                
                st.markdown(f"""
                    <div class='info-card'>
                        <h3 style='color: #1F3C88; font-size: 20px;'>{summary}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                for note in notes_list:
                    st.markdown(f"""
                        <div class='analysis-note'>
                            {note}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                if show_recommendations:
                    st.markdown("## 👨‍⚕️ Clinical Recommendations")
                    recommendations = get_clinical_recommendations(icm, te, exp)
                    
                    rec_html = "<div class='recommendation-box'>"
                    for rec in recommendations:
                        if rec.startswith('###'):
                            clean_heading = rec.replace('###', '').strip()
                            rec_html += f"<h4 style='color: #1F3C88; font-size: 18px; font-weight: 700; margin-top: 15px;'>{clean_heading}</h4>"
                        else:
                            rec_html += f"<p style='color: #2c3e50; font-size: 15px; line-height: 1.8; margin: 8px 0;'>{rec}</p>"
                    rec_html += "</div>"
                    
                    st.markdown(rec_html, unsafe_allow_html=True)
                
                # Download Reports
                st.markdown("## 📥 Download Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Text Report
                    text_report = generate_text_report(icm, te, exp, patient_data)
                    st.download_button(
                        label="📄 Download Text Report",
                        data=text_report,
                        file_name=f"Report_{embryo_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # JSON Export
                    json_data = {
                        'timestamp': timestamp,
                        'patient_data': patient_data,
                        'results': {'icm': icm, 'te': te, 'exp': exp, 'avg': float(avg_score), 'quality': quality_level},
                        'metrics': metrics
                    }
                    st.download_button(
                        label="💾 Export JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"Data_{embryo_id}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

elif page == "📜 History":
    st.markdown("## 📜 Analysis History")
    
    if not st.session_state.history:
        st.info("📭 No history yet. Start analyzing embryos!")
    else:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            search = st.text_input("🔍 Search", placeholder="Patient ID, Embryo ID...")
        
        with col2:
            quality_filter = st.selectbox("Filter", ["All", "Excellent (≥4)", "Moderate (3-4)", "Poor (<3)"])
        
        filtered = st.session_state.history.copy()
        
        if search:
            filtered = [h for h in filtered if search.lower() in str(h['patient_data']).lower()]
        
        if quality_filter != "All":
            if "Excellent" in quality_filter:
                filtered = [h for h in filtered if h['avg_score'] >= 4.0]
            elif "Moderate" in quality_filter:
                filtered = [h for h in filtered if 3.0 <= h['avg_score'] < 4.0]
            else:
                filtered = [h for h in filtered if h['avg_score'] < 3.0]
        
        st.markdown(f"**Showing {len(filtered)} of {len(st.session_state.history)} records**")
        
        for idx, record in enumerate(filtered):
            with st.expander(f"📋 {record['timestamp']} | {record['patient_data']['Embryo ID']} | {record['avg_score']:.2f}/5"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Patient Info:**")
                    for key, value in record['patient_data'].items():
                        if key != "Notes":
                            st.write(f"• {key}: {value}")
                
                with col2:
                    st.markdown("**Grades:**")
                    st.write(f"• ICM: {record['icm']}/5 {'⭐' * record['icm']}")
                    st.write(f"• TE: {record['te']}/5 {'⭐' * record['te']}")
                    st.write(f"• EXP: {record['exp']}/5 {'⭐' * record['exp']}")
        
        if st.button("🗑️ Clear All History"):
            st.session_state.history = []
            st.rerun()

elif page == "📊 Statistics":
    st.markdown("## 📊 Statistics")
    
    if not st.session_state.history:
        st.info("No data yet")
    else:
        total = len(st.session_state.history)
        avg_icm = np.mean([h['icm'] for h in st.session_state.history])
        avg_te = np.mean([h['te'] for h in st.session_state.history])
        avg_exp = np.mean([h['exp'] for h in st.session_state.history])
        
        excellent = sum(1 for h in st.session_state.history if h['avg_score'] >= 4.0)
        moderate = sum(1 for h in st.session_state.history if 3.0 <= h['avg_score'] < 4.0)
        poor = sum(1 for h in st.session_state.history if h['avg_score'] < 3.0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", total)
        col2.metric("Avg ICM", f"{avg_icm:.2f}")
        col3.metric("Avg TE", f"{avg_te:.2f}")
        col4.metric("Avg EXP", f"{avg_exp:.2f}")
        
        st.markdown("### Quality Distribution")
        col1, col2, col3 = st.columns(3)
        col1.metric("Excellent", excellent, f"{excellent/total*100:.1f}%")
        col2.metric("Moderate", moderate, f"{moderate/total*100:.1f}%")
        col3.metric("Poor", poor, f"{poor/total*100:.1f}%")

else:  # About
    st.markdown("## ℹ️ About")
    
    st.markdown("""
    <div class='info-card'>
        <h2>🧬 Blastocyst Grading System v2.0</h2>
        <p>AI-powered embryo quality assessment for IVF clinics.</p>
        
        <h3>Features:</h3>
        <ul>
            <li>Automated ICM, TE, and EXP grading</li>
            <li>Success probability estimation</li>
            <li>Clinical recommendations</li>
            <li>History tracking & statistics</li>
            <li>Multiple export formats</li>
        </ul>
        
        <h3>⚠️ Disclaimer</h3>
        <p style='color: #d32f2f;'>This is an AI-assisted tool. All results must be 
        validated by qualified embryologists and medical professionals.</p>
    </div>
    """, unsafe_allow_html=True)


# Logo
def add_logo():
    try:
        with open("logo.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='position: fixed; right: 25px; bottom: 25px; z-index: 9999;'>
                <img src='data:image/png;base64,{encoded}' style='width: 150px; opacity: 0.9; border-radius: 10px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

add_logo()