import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Blastocyst Grading System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None


#------------Inferences------------
def get_inference(icm, te, exp):
    notes = []

    # ICM (Inner Cell Mass)
    if icm >= 4:
        notes.append("🟢 **ICM:** Good inner cell mass – likely healthy embryoblast.")
    elif icm == 3:
        notes.append("🟡 **ICM:** Moderate ICM – acceptable but not ideal.")
    else:
        notes.append("🔴 **ICM:** Poor ICM – lower likelihood of successful implantation.")

    # TE (Trophectoderm)
    if te >= 4:
        notes.append("🟢 **TE:** Strong trophectoderm – better implantation probability.")
    elif te == 3:
        notes.append("🟡 **TE:** Average TE – may still be viable.")
    else:
        notes.append("🔴 **TE:** Weak TE – may reduce implantation chances.")

    # EXP (Expansion grade)
    if exp >= 4:
        notes.append("🟢 **Expansion:** Well-expanded blastocyst – good development.")
    elif exp == 3:
        notes.append("🟡 **Expansion:** Moderate expansion – monitor carefully.")
    else:
        notes.append("🔴 **Expansion:** Poor expansion – embryo may be underdeveloped.")

    # Overall summary
    if icm >= 4 and te >= 4 and exp >= 4:
        summary = "✅ **Overall:** High-quality blastocyst – strong transfer candidate."
    elif icm >= 3 and te >= 3 and exp >= 3:
        summary = "⚠️ **Overall:** Medium-quality embryo – possible, but requires clinical judgement."
    else:
        summary = "❌ **Overall:** Low-quality blastocyst – poor prognosis."

    return summary, notes


def get_doctor_suggestions(icm, te, exp):
    """Generate detailed doctor suggestions based on grades"""
    suggestions = []
    
    avg_score = (icm + te + exp) / 3
    
    if avg_score >= 4:
        suggestions.append("**Recommended Actions:**")
        suggestions.append("• This is an excellent quality embryo suitable for fresh transfer")
        suggestions.append("• Consider this embryo as a priority for single embryo transfer (SET)")
        suggestions.append("• High probability of successful implantation and pregnancy")
        suggestions.append("• Standard endometrial preparation protocol is recommended")
    elif avg_score >= 3:
        suggestions.append("**Recommended Actions:**")
        suggestions.append("• This embryo has moderate quality and may be considered for transfer")
        suggestions.append("• Discuss with patient about expectations and success rates")
        suggestions.append("• Consider timing of transfer with optimal endometrial receptivity")
        suggestions.append("• May benefit from extended culture or blastocyst monitoring")
        suggestions.append("• Consider PGT-A testing if available and appropriate")
    else:
        suggestions.append("**Recommended Actions:**")
        suggestions.append("• This embryo shows lower quality indicators")
        suggestions.append("• Consider extended culture to day 6 to assess further development")
        suggestions.append("• Discuss alternative options with the patient:")
        suggestions.append("  - Banking additional embryos from a new cycle")
        suggestions.append("  - Considering embryo pooling strategy")
        suggestions.append("  - Reviewing stimulation protocol for future cycles")
        suggestions.append("• If transfer is attempted, manage patient expectations appropriately")
        suggestions.append("• Consider comprehensive fertility evaluation")
    
    suggestions.append("\n**Parameter-Specific Recommendations:**")
    
    if icm < 3:
        suggestions.append("• **ICM:** Poor inner cell mass may indicate:")
        suggestions.append("  - Suboptimal culture conditions - review lab protocols")
        suggestions.append("  - Consider supplementation with growth factors in culture media")
        suggestions.append("  - Evaluate oocyte quality and maturation protocols")
    
    if te < 3:
        suggestions.append("• **TE:** Weak trophectoderm suggests:")
        suggestions.append("  - Potential implantation challenges")
        suggestions.append("  - Consider assisted hatching if transfer proceeds")
        suggestions.append("  - Optimize progesterone support for endometrial receptivity")
    
    if exp < 3:
        suggestions.append("• **EXP:** Poor expansion indicates:")
        suggestions.append("  - Embryo may benefit from additional culture time")
        suggestions.append("  - Review zona pellucida thickness")
        suggestions.append("  - Consider vitrification for later transfer if appropriate")
    
    suggestions.append("\n**Follow-up Recommendations:**")
    suggestions.append("• Schedule detailed consultation with patient")
    suggestions.append("• Document findings in patient medical record")
    suggestions.append("• Consider counseling session to discuss outcomes and next steps")
    
    return suggestions


def calculate_success_probability(icm, te, exp):
    """Calculate estimated success probability based on grades"""
    # Weighted formula based on clinical research
    base_score = (icm * 0.4 + te * 0.35 + exp * 0.25) / 5 * 100
    
    if base_score >= 80:
        return f"{base_score:.1f}% - Excellent", "success"
    elif base_score >= 60:
        return f"{base_score:.1f}% - Good", "warning"
    elif base_score >= 40:
        return f"{base_score:.1f}% - Fair", "info"
    else:
        return f"{base_score:.1f}% - Poor", "error"


def generate_pdf_report(icm, te, exp, img_array, patient_info=None, image_path=None):
    """Generate a professional PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1F3C88'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1F3C88'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    # Title
    story.append(Paragraph("🧬 BLASTOCYST GRADING REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report info
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
    story.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", body_style))
    story.append(Paragraph("<b>Analysis System:</b> AI-Powered Embryo Grading System v2.0", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    if patient_info:
        story.append(Paragraph("PATIENT INFORMATION", heading_style))
        patient_data = [[k, v] for k, v in patient_info.items()]
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F0FE')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Grading Results
    story.append(Paragraph("GRADING RESULTS", heading_style))
    
    avg_score = (icm + te + exp) / 3
    success_prob, _ = calculate_success_probability(icm, te, exp)
    
    grading_data = [
        ['Parameter', 'Grade', 'Score'],
        ['Inner Cell Mass (ICM)', f'{icm}/5', '⭐' * icm],
        ['Trophectoderm (TE)', f'{te}/5', '⭐' * te],
        ['Expansion (EXP)', f'{exp}/5', '⭐' * exp],
        ['Average Score', f'{avg_score:.2f}/5', ''],
        ['Success Probability', success_prob, '']
    ]
    
    grading_table = Table(grading_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    grading_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3C88')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(grading_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Analysis
    story.append(Paragraph("DETAILED ANALYSIS", heading_style))
    summary, notes = get_inference(icm, te, exp)
    
    clean_summary = summary.replace("**", "").replace("✅", "").replace("⚠️", "").replace("❌", "")
    story.append(Paragraph(clean_summary, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    for note in notes:
        clean_note = note.replace("**", "<b>").replace("**", "</b>").replace("🟢", "✓").replace("🟡", "⚠").replace("🔴", "✗")
        story.append(Paragraph(clean_note, body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Clinical Recommendations
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    suggestions = get_doctor_suggestions(icm, te, exp)
    
    for suggestion in suggestions:
        clean_suggestion = suggestion.replace("**", "<b>").replace("**", "</b>")
        story.append(Paragraph(clean_suggestion, body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Grading Criteria Reference
    story.append(Paragraph("GRADING CRITERIA REFERENCE", heading_style))
    
    criteria_text = """
    <b>Inner Cell Mass (ICM):</b><br/>
    Grade 5: Many cells, tightly packed<br/>
    Grade 4: Several cells, loosely grouped<br/>
    Grade 3: Few cells<br/>
    Grade 2: Very few cells<br/>
    Grade 1: Poor, degenerating cells<br/><br/>
    
    <b>Trophectoderm (TE):</b><br/>
    Grade 5: Many cells forming cohesive epithelium<br/>
    Grade 4: Several cells forming loose epithelium<br/>
    Grade 3: Few cells<br/>
    Grade 2: Very few cells<br/>
    Grade 1: Poor, few large cells<br/><br/>
    
    <b>Expansion (EXP):</b><br/>
    Grade 5: Hatching/hatched blastocyst<br/>
    Grade 4: Expanded, thin zona pellucida<br/>
    Grade 3: Full blastocyst, thick zona<br/>
    Grade 2: Early blastocyst, small cavity<br/>
    Grade 1: Early blastocyst, barely visible cavity
    """
    story.append(Paragraph(criteria_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Paragraph("DISCLAIMER", heading_style))
    disclaimer_text = """
    This report is generated by an AI-assisted analysis system and should be reviewed and 
    validated by qualified embryologists and fertility specialists. Clinical decisions should 
    be made in conjunction with comprehensive patient evaluation and professional medical judgment.
    """
    story.append(Paragraph(disclaimer_text, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def save_to_history(patient_info, icm, te, exp, timestamp):
    """Save analysis to history"""
    history_entry = {
        'timestamp': timestamp,
        'patient_info': patient_info,
        'icm': icm,
        'te': te,
        'exp': exp,
        'avg_score': (icm + te + exp) / 3
    }
    st.session_state.history.insert(0, history_entry)
    # Keep only last 50 entries
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]


IMG_SIZE = 224

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Center content */
    .center-content {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Card styling */
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .suggestion-box {
        background: rgba(255, 248, 220, 0.95);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FFA500;
        margin: 15px 0;
    }
    
    /* Button styling */
    .stDownloadButton button {
        width: 100%;
        background: linear-gradient(90deg, #1F3C88 0%, #2E5BFF 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px;
        border: none;
    }
    
    /* History item styling */
    .history-item {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1F3C88;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Stats box */
    .stats-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


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

add_bg("background1.jpg")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("embryo_output_model.keras")

model = load_model()


# Sidebar
with st.sidebar:
    st.markdown("### 🧬 Navigation")
    page = st.radio("Go to", ["📊 Analysis", "📜 History", "📈 Statistics", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_probabilities = st.checkbox("Show Success Probabilities", value=True)
    detailed_mode = st.checkbox("Detailed Analysis Mode", value=True)
    
    st.markdown("---")
    st.markdown("### 📞 Support")
    st.info("For technical support, contact: support@embryograding.com")


# Main content based on page selection
if page == "📊 Analysis":
    # Header
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1 style='color:#1F3C88; text-shadow:0px 2px 6px rgba(0,0,0,0.25);'>
                🧬 Blastocyst Grading System
            </h1>
            <p style='color:#666; font-size: 18px;'>AI-Powered Embryo Quality Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Patient Information Section
    with st.expander("📋 Patient Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="e.g., PT-2024-001")
            patient_name = st.text_input("Patient Name", placeholder="Optional")
        with col2:
            patient_age = st.number_input("Age", min_value=18, max_value=50, value=30)
            doctor_name = st.text_input("Doctor Name", placeholder="Optional")
        with col3:
            cycle_day = st.selectbox("Cycle Day", ["Day 3", "Day 5", "Day 6"])
            embryo_id = st.text_input("Embryo ID", placeholder="e.g., EMB-001")
    
    # File upload
    st.markdown("### 📤 Upload Embryo Image")
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
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
    
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded Embryo Image", use_container_width=True)
        
        # Analyze button
        if st.button("🔬 Analyze Embryo", type="primary", use_container_width=True):
            with st.spinner("Analyzing embryo image..."):
                icm, te, exp = predict(img)
                
                # Prepare patient info
                patient_info = {}
                if patient_id:
                    patient_info["Patient ID"] = patient_id
                if patient_name:
                    patient_info["Patient Name"] = patient_name
                if patient_age:
                    patient_info["Age"] = str(patient_age)
                if doctor_name:
                    patient_info["Doctor"] = doctor_name
                if cycle_day:
                    patient_info["Cycle Day"] = cycle_day
                if embryo_id:
                    patient_info["Embryo ID"] = embryo_id
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save to history
                save_to_history(patient_info, icm, te, exp, timestamp)
                
                st.success("✅ Analysis Complete!")
                
                # Results section
                st.markdown("---")
                st.markdown("## 📊 Grading Results")
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>ICM</h3>
                            <h1>{icm}/5</h1>
                            <p>{'⭐' * icm}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>TE</h3>
                            <h1>{te}/5</h1>
                            <p>{'⭐' * te}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>EXP</h3>
                            <h1>{exp}/5</h1>
                            <p>{'⭐' * exp}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    avg_score = (icm + te + exp) / 3
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Average</h3>
                            <h1>{avg_score:.2f}</h1>
                            <p>Overall Score</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Success probability
                if show_probabilities:
                    success_prob, prob_type = calculate_success_probability(icm, te, exp)
                    st.markdown(f"""
                        <div class='info-card' style='text-align: center;'>
                            <h3>🎯 Estimated Success Probability</h3>
                            <h2 style='color: #1F3C88;'>{success_prob}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Inference
                st.markdown("## 🔬 Clinical Inference")
                summary, notes = get_inference(icm, te, exp)
                
                st.markdown(f"""
                    <div class='info-card'>
                        <h3>{summary}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                for note in notes:
                    st.markdown(f"<div style='padding: 8px;'>{note}</div>", unsafe_allow_html=True)
                
                # Recommendations
                if detailed_mode:
                    st.markdown("## 👨‍⚕️ Clinical Recommendations")
                    suggestions = get_doctor_suggestions(icm, te, exp)
                    
                    suggestions_html = "<div class='suggestion-box'>"
                    for suggestion in suggestions:
                        suggestions_html += f"<p style='margin: 6px 0;'>{suggestion}</p>"
                    suggestions_html += "</div>"
                    
                    st.markdown(suggestions_html, unsafe_allow_html=True)
                
                # Download section
                st.markdown("## 📥 Download Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    pdf_buffer = generate_pdf_report(icm, te, exp, img, patient_info)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"blastocyst_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with col2:
                    # JSON export
                    json_data = {
                        'timestamp': timestamp,
                        'patient_info': patient_info,
                        'results': {
                            'icm': icm,
                            'te': te,
                            'exp': exp,
                            'average': float(avg_score)
                        }
                    }
                    st.download_button(
                        label="💾 Export as JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"embryo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

elif page == "📜 History":
    st.markdown("## 📜 Analysis History")
    
    if len(st.session_state.history) == 0:
        st.info("No analysis history yet. Start by analyzing an embryo image!")
    else:
        # Search and filter
        search_term = st.text_input("🔍 Search by Patient ID or Embryo ID", "")
        
        filtered_history = st.session_state.history
        if search_term:
            filtered_history = [
                h for h in st.session_state.history 
                if search_term.lower() in str(h.get('patient_info', {})).lower()
            ]
        
        st.markdown(f"**Total Records:** {len(filtered_history)}")
        
        # Display history
        for idx, entry in enumerate(filtered_history):
            with st.expander(f"📋 {entry['timestamp']} - Avg Score: {entry['avg_score']:.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Patient Information:**")
                    for key, value in entry['patient_info'].items():
                        st.write(f"- {key}: {value}")
                
                with col2:
                    st.markdown("**Grading Results:**")
                    st.write(f"- ICM: {entry['icm']}/5")
                    st.write(f"- TE: {entry['te']}/5")
                    st.write(f"- EXP: {entry['exp']}/5")
                    st.write(f"- Average: {entry['avg_score']:.2f}/5")
        
        # Clear history button
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.history = []
            st.rerun()

elif page == "📈 Statistics":
    st.markdown("## 📈 Analysis Statistics")
    
    if len(st.session_state.history) == 0:
        st.info("No data available yet. Analyze some embryos to see statistics!")
    else:
        # Calculate statistics
        total_analyses = len(st.session_state.history)
        avg_icm = np.mean([h['icm'] for h in st.session_state.history])
        avg_te = np.mean([h['te'] for h in st.session_state.history])
        avg_exp = np.mean([h['exp'] for h in st.session_state.history])
        avg_overall = np.mean([h['avg_score'] for h in st.session_state.history])
        
        high_quality = sum(1 for h in st.session_state.history if h['avg_score'] >= 4)
        medium_quality = sum(1 for h in st.session_state.history if 3 <= h['avg_score'] < 4)
        low_quality = sum(1 for h in st.session_state.history if h['avg_score'] < 3)
        
        # Display stats in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='stats-box'>
                    <h3>Total Analyses</h3>
                    <h1>{total_analyses}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='stats-box'>
                    <h3>Avg ICM</h3>
                    <h1>{avg_icm:.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='stats-box'>
                    <h3>Avg TE</h3>
                    <h1>{avg_te:.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='stats-box'>
                    <h3>Avg EXP</h3>
                    <h1>{avg_exp:.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quality distribution
        st.markdown("### 📊 Quality Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Quality (≥4.0)", high_quality, 
                     f"{(high_quality/total_analyses*100):.1f}%")
        
        with col2:
            st.metric("Medium Quality (3.0-3.9)", medium_quality,
                     f"{(medium_quality/total_analyses*100):.1f}%")
        
        with col3:
            st.metric("Low Quality (<3.0)", low_quality,
                     f"{(low_quality/total_analyses*100):.1f}%")
        
        # Recent trends
        st.markdown("### 📈 Recent Trends")
        if len(st.session_state.history) >= 5:
            recent_5 = st.session_state.history[:5]
            recent_avg = np.mean([h['avg_score'] for h in recent_5])
            st.write(f"**Last 5 Analyses Average:** {recent_avg:.2f}/5")
        
        # Export statistics
        if st.button("📊 Export Statistics Report"):
            stats_data = {
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_analyses': total_analyses,
                'averages': {
                    'icm': float(avg_icm),
                    'te': float(avg_te),
                    'exp': float(avg_exp),
                    'overall': float(avg_overall)
                },
                'distribution': {
                    'high_quality': high_quality,
                    'medium_quality': medium_quality,
                    'low_quality': low_quality
                }
            }
            st.download_button(
                label="Download Statistics JSON",
                data=json.dumps(stats_data, indent=2),
                file_name=f"statistics_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

else:  # About page
    st.markdown("## ℹ️ About the System")
    
    st.markdown("""
    <div class='info-card'>
        <h3>🧬 Blastocyst Grading System</h3>
        <p>This AI-powered system provides automated assessment of blastocyst quality 
        for in-vitro fertilization (IVF) procedures.</p>
        
        <h4>Key Features:</h4>
        <ul>
            <li>✅ Automated grading of ICM, TE, and Expansion</li>
            <li>📊 Real-time success probability estimation</li>
            <li>📄 Professional PDF report generation</li>
            <li>📜 Complete analysis history tracking</li>
            <li>📈 Statistical analysis and trends</li>
            <li>💾 JSON export for data integration</li>
        </ul>
        
        <h4>Grading System:</h4>
        <p>The system evaluates three critical parameters:</p>
        <ul>
            <li><b>ICM (Inner Cell Mass):</b> Quality of cells that form the fetus</li>
            <li><b>TE (Trophectoderm):</b> Quality of cells that form the placenta</li>
            <li><b>EXP (Expansion):</b> Development stage of the blastocyst</li>
        </ul>
        
        <h4>Version Information:</h4>
        <p>Version 2.0 - Enhanced with history tracking and PDF reports</p>
        
        <h4>Disclaimer:</h4>
        <p style='color: #d32f2f;'><b>Important:</b> This system is designed to assist 
        qualified medical professionals. All results should be verified by certified 
        embryologists. Clinical decisions must be made by qualified healthcare providers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Grading Reference")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **ICM Grades:**
        - **5:** Many cells, tightly packed
        - **4:** Several cells, loosely grouped
        - **3:** Few cells
        - **2:** Very few cells
        - **1:** Poor, degenerating
        """)
    
    with col2:
        st.markdown("""
        **TE Grades:**
        - **5:** Cohesive epithelium
        - **4:** Loose epithelium
        - **3:** Few cells
        - **2:** Very few cells
        - **1:** Poor, large cells
        """)
    
    with col3:
        st.markdown("""
        **EXP Grades:**
        - **5:** Hatching/hatched
        - **4:** Expanded, thin zona
        - **3:** Full, thick zona
        - **2:** Early, small cavity
        - **1:** Barely visible cavity
        """)


# Footer
def add_logo():
    try:
        with open("logo.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='position: fixed; right: 25px; bottom: 25px; z-index: 9999;'>
                <img src='data:image/png;base64,{encoded}' style='width: 180px; opacity: 0.92;'>
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

add_logo()