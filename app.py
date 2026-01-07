import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import base64
from datetime import datetime
import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
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

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = True

IMG_SIZE = 224

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
        quality_level = "Excellent"
    elif icm >= 3 and te >= 3 and exp >= 3:
        summary = "⚠️ **Overall:** Medium-quality embryo – possible, but requires clinical judgement."
        quality_level = "Moderate"
    else:
        summary = "❌ **Overall:** Low-quality blastocyst – poor prognosis."
        quality_level = "Poor"

    return summary, notes, quality_level


def get_clinical_recommendations(icm, te, exp):
    """Generate detailed clinical recommendations"""
    recommendations = []
    avg_score = (icm + te + exp) / 3
    
    # Overall recommendations
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
    
    # Specific parameter recommendations
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
        recommendations.append("- Monitor endometrial receptivity carefully")
    
    if exp < 3:
        recommendations.append("**Expansion Concerns:**")
        recommendations.append("- Embryo may benefit from additional 12-24 hours culture")
        recommendations.append("- Assess zona pellucida thickness")
        recommendations.append("- Consider delayed transfer protocol")
    
    # Follow-up actions
    recommendations.append("\n### 📋 Follow-Up Actions")
    recommendations.append("- Schedule detailed consultation with reproductive endocrinologist")
    recommendations.append("- Document findings in comprehensive medical record")
    recommendations.append("- Discuss realistic expectations with patient")
    recommendations.append("- Plan for potential additional cycles if needed")
    recommendations.append("- Consider PGT-A testing for future embryos if indicated")
    
    return recommendations


def calculate_success_metrics(icm, te, exp):
    """Calculate various success metrics"""
    avg_score = (icm + te + exp) / 3
    
    # Implantation probability (weighted formula based on clinical data)
    base_prob = (icm * 0.35 + te * 0.40 + exp * 0.25) / 5 * 100
    
    # Adjust based on combinations
    if icm >= 4 and te >= 4:
        base_prob += 10
    if exp == 5:  # Hatching embryo bonus
        base_prob += 5
    
    base_prob = min(base_prob, 95)  # Cap at 95%
    
    # Quality score
    quality_score = f"{avg_score:.2f}/5.00"
    
    # Development stage
    if exp >= 4:
        stage = "Advanced Development"
    elif exp == 3:
        stage = "Standard Development"
    else:
        stage = "Early Development"
    
    return {
        'success_probability': f"{base_prob:.1f}%",
        'quality_score': quality_score,
        'development_stage': stage,
        'transfer_priority': 'High' if avg_score >= 4 else 'Medium' if avg_score >= 3 else 'Low'
    }


def generate_pdf_report(icm, te, exp, patient_data, img_array=None):
    """Generate comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1F3C88'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1F3C88'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14
    )
    
    # Header
    story.append(Paragraph("🧬 BLASTOCYST GRADING REPORT", title_style))
    story.append(Paragraph("AI-Powered Embryo Quality Assessment", 
                          ParagraphStyle('Subtitle', parent=body_style, alignment=TA_CENTER, 
                                       fontSize=11, textColor=colors.grey)))
    story.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", body_style))
    story.append(Paragraph(f"<b>Report ID:</b> {datetime.now().strftime('%Y%m%d-%H%M%S')}", body_style))
    story.append(Paragraph("<b>System Version:</b> v2.0", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient/Embryo Information
    story.append(Paragraph("PATIENT & EMBRYO INFORMATION", heading_style))
    
    info_data = []
    for key, value in patient_data.items():
        if value:  # Only include non-empty fields
            info_data.append([key, str(value)])
    
    if info_data:
        info_table = Table(info_data, colWidths=[2.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F4FF')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(info_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Grading Results
    story.append(Paragraph("GRADING RESULTS", heading_style))
    
    metrics = calculate_success_metrics(icm, te, exp)
    
    grading_data = [
        ['Parameter', 'Grade', 'Assessment'],
        ['Inner Cell Mass (ICM)', f'{icm}/5', '⭐' * icm + '☆' * (5-icm)],
        ['Trophectoderm (TE)', f'{te}/5', '⭐' * te + '☆' * (5-te)],
        ['Expansion Grade (EXP)', f'{exp}/5', '⭐' * exp + '☆' * (5-exp)],
    ]
    
    grading_table = Table(grading_data, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
    grading_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3C88')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFEF0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(grading_table)
    story.append(Spacer(1, 0.15*inch))
    
    # Success Metrics
    metrics_data = [
        ['Metric', 'Value'],
        ['Quality Score', metrics['quality_score']],
        ['Success Probability', metrics['success_probability']],
        ['Development Stage', metrics['development_stage']],
        ['Transfer Priority', metrics['transfer_priority']],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F1F8F4')),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Clinical Analysis
    story.append(Paragraph("CLINICAL ANALYSIS", heading_style))
    summary, notes, quality_level = get_inference(icm, te, exp)
    
    clean_summary = summary.replace('**', '').replace('✅', '').replace('⚠️', '').replace('❌', '')
    story.append(Paragraph(f"<b>Overall Assessment:</b> {clean_summary}", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    for note in notes:
        clean_note = note.replace('**', '<b>').replace('</b>', '</b>').replace('🟢', '✓ ').replace('🟡', '⚠ ').replace('🔴', '✗ ')
        story.append(Paragraph(clean_note, body_style))
    
    story.append(Spacer(1, 0.25*inch))
    
    # Clinical Recommendations
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    recommendations = get_clinical_recommendations(icm, te, exp)
    
    for rec in recommendations:
        if rec.startswith('###'):
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(rec.replace('### ', '').replace('🎯', '').replace('🔬', '').replace('📋', ''), 
                                 ParagraphStyle('SubHeading', parent=body_style, fontSize=11, 
                                              fontName='Helvetica-Bold', textColor=colors.HexColor('#2E5BFF'))))
        else:
            clean_rec = rec.replace('**', '<b>').replace('**', '</b>')
            story.append(Paragraph(clean_rec, body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Grading Reference
    story.append(Paragraph("GRADING CRITERIA REFERENCE", heading_style))
    
    criteria = """
    <b>Inner Cell Mass (ICM) Grading Scale:</b><br/>
    • Grade 5: Many cells, tightly packed (Excellent)<br/>
    • Grade 4: Several cells, loosely grouped (Very Good)<br/>
    • Grade 3: Few cells (Good)<br/>
    • Grade 2: Very few cells (Fair)<br/>
    • Grade 1: Poor, degenerating cells (Poor)<br/><br/>
    
    <b>Trophectoderm (TE) Grading Scale:</b><br/>
    • Grade 5: Many cells forming cohesive epithelium (Excellent)<br/>
    • Grade 4: Several cells forming loose epithelium (Very Good)<br/>
    • Grade 3: Few cells (Good)<br/>
    • Grade 2: Very few cells (Fair)<br/>
    • Grade 1: Poor, few large cells (Poor)<br/><br/>
    
    <b>Expansion (EXP) Grading Scale:</b><br/>
    • Grade 5: Hatching or fully hatched blastocyst (Excellent)<br/>
    • Grade 4: Expanded blastocyst, thin zona pellucida (Very Good)<br/>
    • Grade 3: Full blastocyst, thick zona pellucida (Good)<br/>
    • Grade 2: Early blastocyst, small blastocoel cavity (Fair)<br/>
    • Grade 1: Early blastocyst, barely visible cavity (Poor)
    """
    story.append(Paragraph(criteria, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Paragraph("MEDICAL DISCLAIMER", heading_style))
    disclaimer = """
    This report is generated by an AI-assisted embryo grading system designed to support clinical 
    decision-making. All results must be reviewed and validated by qualified embryologists and 
    reproductive medicine specialists. Clinical decisions should be made based on comprehensive 
    patient evaluation, medical history, and professional medical judgment. This report should not 
    be used as the sole basis for treatment decisions. The success probabilities presented are 
    estimates based on general statistical data and may not reflect individual patient outcomes.
    """
    story.append(Paragraph(disclaimer, ParagraphStyle('Disclaimer', parent=body_style, 
                                                     fontSize=9, textColor=colors.grey)))
    
    # Footer
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<i>Report generated by Blastocyst Grading System v2.0 | {timestamp}</i>", 
                          ParagraphStyle('Footer', parent=body_style, fontSize=8, 
                                       alignment=TA_CENTER, textColor=colors.grey)))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def save_to_history(analysis_data):
    """Save analysis to session history"""
    st.session_state.history.insert(0, analysis_data)
    # Keep last 100 records
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[:100]


# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
    }
    
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
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #FFF5E1 0%, #FFE4B5 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FFA500;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .history-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #1F3C88;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    
    .center-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
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

add_bg("background1.jpg")


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
    st.markdown("## 🧬 System Control")
    
    page = st.selectbox("Navigate", 
                       ["🔬 Analysis", "📜 History", "📊 Statistics", "ℹ️ About"],
                       label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_recommendations = st.checkbox("Show Clinical Recommendations", value=True)
    show_metrics = st.checkbox("Show Success Metrics", value=True)
    auto_save = st.checkbox("Auto-save to History", value=True)
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if st.session_state.history:
        st.metric("Total Analyses", len(st.session_state.history))
        avg_quality = np.mean([h['avg_score'] for h in st.session_state.history])
        st.metric("Avg Quality Score", f"{avg_quality:.2f}/5")
    else:
        st.info("No analyses yet")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("Upload clear, well-focused embryo images for best results")


# Main Content
if page == "🔬 Analysis":
    # Header
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1 style='color:#1F3C88; text-shadow:0px 2px 6px rgba(0,0,0,0.25);'>
                🧬 Blastocyst Grading System
            </h1>
            <p style='color:#666; font-size: 18px; font-weight: 500;'>
                AI-Powered Embryo Quality Assessment Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Embryo Details Form
    with st.expander("📋 Embryo & Patient Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="e.g., PT-2024-001")
            patient_age = st.number_input("Patient Age", min_value=18, max_value=50, value=32)
            clinic_name = st.text_input("Clinic Name", placeholder="Optional")
        
        with col2:
            embryo_id = st.text_input("Embryo ID", placeholder="e.g., EMB-001")
            cycle_day = st.selectbox("Development Day", ["Day 3", "Day 5", "Day 6"], index=1)
            collection_date = st.date_input("Collection Date", value=datetime.now())
        
        with col3:
            doctor_name = st.text_input("Embryologist", placeholder="Name")
            culture_media = st.text_input("Culture Media", placeholder="Optional")
            notes = st.text_area("Additional Notes", placeholder="Any observations...", height=100)
    
    # File Upload
    st.markdown("### 📤 Upload Embryo Image")
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded Embryo Image", use_container_width=True)
        
        # Analyze Button
        if st.button("🔬 Analyze Embryo", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing embryo image..."):
                icm, te, exp = predict(img)
                
                # Prepare data
                patient_data = {
                    "Patient ID": patient_id if patient_id else "N/A",
                    "Patient Age": patient_age,
                    "Embryo ID": embryo_id if embryo_id else "N/A",
                    "Development Day": cycle_day,
                    "Collection Date": collection_date.strftime("%Y-%m-%d"),
                    "Embryologist": doctor_name if doctor_name else "N/A",
                    "Clinic": clinic_name if clinic_name else "N/A",
                    "Culture Media": culture_media if culture_media else "N/A",
                    "Notes": notes if notes else "None"
                }
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                avg_score = (icm + te + exp) / 3
                
                # Save to history
                if auto_save:
                    analysis_record = {
                        'timestamp': timestamp,
                        'patient_data': patient_data,
                        'icm': icm,
                        'te': te,
                        'exp': exp,
                        'avg_score': avg_score
                    }
                    save_to_history(analysis_record)
                
                st.success("✅ Analysis Complete!")
                
                # Results Display
                st.markdown("---")
                st.markdown("## 📊 Grading Results")
                
                # Metrics Cards
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
                            <p>Quality Score</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Success Metrics
                if show_metrics:
                    metrics = calculate_success_metrics(icm, te, exp)
                    
                    st.markdown(f"""
                        <div class='success-prob'>
                            🎯 Estimated Success Probability: {metrics['success_probability']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Development Stage", metrics['development_stage'])
                    col2.metric("Transfer Priority", metrics['transfer_priority'])
                    col3.metric("Quality Score", metrics['quality_score'])
                
                # Clinical Analysis
                st.markdown("## 🔬 Clinical Analysis")
                summary, notes_list, quality_level = get_inference(icm, te, exp)
                
                st.markdown(f"""
                    <div class='info-card'>
                        <h3>{summary}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                for note in notes_list:
                    st.markdown(f"<div style='padding: 10px; font-size: 16px;'>{note}</div>", 
                              unsafe_allow_html=True)
                
                # Clinical Recommendations
                if show_recommendations:
                    st.markdown("## 👨‍⚕️ Clinical Recommendations")
                    recommendations = get_clinical_recommendations(icm, te, exp)
                    
                    rec_html = "<div class='recommendation-box'>"
                    for rec in recommendations:
                        if rec.startswith('###'):
                            rec_html += f"<h4 style='color: #1F3C88; margin-top: 15px;'>{rec.replace('###', '').strip()}</h4>"
                        else:
                            rec_html += f"<p style='margin: 5px 0; line-height: 1.6;'>{rec}</p>"
                    rec_html += "</div>"
                    
                    st.markdown(rec_html, unsafe_allow_html=True)
                
                # Download Reports
                st.markdown("## 📥 Download Reports")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # PDF Report
                    pdf_buffer = generate_pdf_report(icm, te, exp, patient_data, img)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"Embryo_Report_{embryo_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with col2:
                    # JSON Export
                    json_data = {
                        'timestamp': timestamp,
                        'patient_data': patient_data,
                        'results': {
                            'icm': icm,
                            'te': te,
                            'exp': exp,
                            'average_score': float(avg_score),
                            'quality_level': quality_level
                        },
                        'metrics': metrics
                    }
                    st.download_button(
                        label="💾 Export JSON Data",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"Embryo_Data_{embryo_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # CSV Summary
                    csv_data = f"Parameter,Value\nICM,{icm}\nTE,{te}\nEXP,{exp}\nAverage,{avg_score:.2f}\nQuality,{quality_level}\nSuccess Probability,{metrics['success_probability']}"
                    st.download_button(
                        label="📊 Export CSV Summary",
                        data=csv_data,
                        file_name=f"Embryo_Summary_{embryo_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

elif page == "📜 History":
    st.markdown("## 📜 Analysis History")
    
    if not st.session_state.history:
        st.info("📭 No analysis history available. Start by analyzing embryo images!")
    else:
        # Search and filters
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            search_query = st.text_input("🔍 Search", placeholder="Patient ID, Embryo ID, etc.")
        
        with col2:
            quality_filter = st.selectbox("Filter by Quality", 
                                         ["All", "Excellent (≥4.0)", "Moderate (3.0-3.9)", "Poor (<3.0)"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Most Recent", "Highest Quality", "Lowest Quality"])
        
        # Filter history
        filtered_history = st.session_state.history.copy()
        
        if search_query:
            filtered_history = [h for h in filtered_history 
                              if search_query.lower() in str(h['patient_data']).lower()]
        
        if quality_filter != "All":
            if "Excellent" in quality_filter:
                filtered_history = [h for h in filtered_history if h['avg_score'] >= 4.0]
            elif "Moderate" in quality_filter:
                filtered_history = [h for h in filtered_history if 3.0 <= h['avg_score'] < 4.0]
            else:
                filtered_history = [h for h in filtered_history if h['avg_score'] < 3.0]
        
        # Sort
        if sort_by == "Highest Quality":
            filtered_history.sort(key=lambda x: x['avg_score'], reverse=True)
        elif sort_by == "Lowest Quality":
            filtered_history.sort(key=lambda x: x['avg_score'])
        
        st.markdown(f"**Showing {len(filtered_history)} of {len(st.session_state.history)} records**")
        
        # Display history
        for idx, record in enumerate(filtered_history):
            with st.expander(f"📋 {record['timestamp']} | {record['patient_data']['Embryo ID']} | Score: {record['avg_score']:.2f}/5"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Patient Information:**")
                    for key, value in record['patient_data'].items():
                        if key != "Notes":
                            st.write(f"• {key}: {value}")
                
                with col2:
                    st.markdown("**Grading Results:**")
                    st.write(f"• ICM: {record['icm']}/5 {'⭐' * record['icm']}")
                    st.write(f"• TE: {record['te']}/5 {'⭐' * record['te']}")
                    st.write(f"• EXP: {record['exp']}/5 {'⭐' * record['exp']}")
                    st.write(f"• Average: {record['avg_score']:.2f}/5")
                
                if record['patient_data'].get('Notes') and record['patient_data']['Notes'] != 'None':
                    st.markdown("**Notes:**")
                    st.write(record['patient_data']['Notes'])
                
                # Re-generate report button
                if st.button(f"📄 Generate Report", key=f"report_{idx}"):
                    pdf_buffer = generate_pdf_report(
                        record['icm'], record['te'], record['exp'], 
                        record['patient_data']
                    )
                    st.download_button(
                        label="Download PDF",
                        data=pdf_buffer,
                        file_name=f"Embryo_Report_{record['patient_data']['Embryo ID']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        key=f"download_{idx}"
                    )
        
        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All History"):
                st.session_state.history = []
                st.rerun()
        
        with col2:
            # Export all history
            all_history_json = json.dumps(st.session_state.history, indent=2, default=str)
            st.download_button(
                label="💾 Export All History",
                data=all_history_json,
                file_name=f"Complete_History_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

elif page == "📊 Statistics":
    st.markdown("## 📊 Analysis Statistics")
    
    if not st.session_state.history:
        st.info("📊 No data available. Analyze embryos to see statistics!")
    else:
        # Calculate statistics
        total = len(st.session_state.history)
        avg_icm = np.mean([h['icm'] for h in st.session_state.history])
        avg_te = np.mean([h['te'] for h in st.session_state.history])
        avg_exp = np.mean([h['exp'] for h in st.session_state.history])
        avg_overall = np.mean([h['avg_score'] for h in st.session_state.history])
        
        excellent = sum(1 for h in st.session_state.history if h['avg_score'] >= 4.0)
        moderate = sum(1 for h in st.session_state.history if 3.0 <= h['avg_score'] < 4.0)
        poor = sum(1 for h in st.session_state.history if h['avg_score'] < 3.0)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Analyses", total)
        col2.metric("Average ICM", f"{avg_icm:.2f}/5")
        col3.metric("Average TE", f"{avg_te:.2f}/5")
        col4.metric("Average EXP", f"{avg_exp:.2f}/5")
        
        st.markdown("---")
        
        # Quality Distribution
        st.markdown("### 📈 Quality Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class='info-card' style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white;'>
                    <h2>{excellent}</h2>
                    <p>Excellent Quality</p>
                    <p style='font-size: 20px;'>{(excellent/total*100):.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='info-card' style='background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); color: white;'>
                    <h2>{moderate}</h2>
                    <p>Moderate Quality</p>
                    <p style='font-size: 20px;'>{(moderate/total*100):.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='info-card' style='background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); color: white;'>
                    <h2>{poor}</h2>
                    <p>Poor Quality</p>
                    <p style='font-size: 20px;'>{(poor/total*100):.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Recent trends
        st.markdown("### 📉 Recent Trends")
        
        if total >= 10:
            recent_10 = st.session_state.history[:10]
            recent_avg = np.mean([h['avg_score'] for h in recent_10])
            older_avg = np.mean([h['avg_score'] for h in st.session_state.history[10:]])
            trend = recent_avg - older_avg
            
            col1, col2 = st.columns(2)
            col1.metric("Last 10 Analyses Avg", f"{recent_avg:.2f}", f"{trend:+.2f}")
            col2.metric("Overall Average", f"{avg_overall:.2f}")
        
        # Grade distribution
        st.markdown("### 🎯 Grade Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**ICM Distribution**")
            for grade in range(5, 0, -1):
                count = sum(1 for h in st.session_state.history if h['icm'] == grade)
                percentage = (count / total * 100) if total > 0 else 0
                st.write(f"Grade {grade}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.markdown("**TE Distribution**")
            for grade in range(5, 0, -1):
                count = sum(1 for h in st.session_state.history if h['te'] == grade)
                percentage = (count / total * 100) if total > 0 else 0
                st.write(f"Grade {grade}: {count} ({percentage:.1f}%)")
        
        with col3:
            st.markdown("**EXP Distribution**")
            for grade in range(5, 0, -1):
                count = sum(1 for h in st.session_state.history if h['exp'] == grade)
                percentage = (count / total * 100) if total > 0 else 0
                st.write(f"Grade {grade}: {count} ({percentage:.1f}%)")
        
        # Export statistics
        st.markdown("---")
        if st.button("📊 Generate Statistics Report"):
            stats_report = {
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_analyses': total,
                'averages': {
                    'icm': float(avg_icm),
                    'te': float(avg_te),
                    'exp': float(avg_exp),
                    'overall': float(avg_overall)
                },
                'quality_distribution': {
                    'excellent': excellent,
                    'moderate': moderate,
                    'poor': poor
                },
                'percentages': {
                    'excellent': float(excellent/total*100),
                    'moderate': float(moderate/total*100),
                    'poor': float(poor/total*100)
                }
            }
            
            st.download_button(
                label="Download Statistics JSON",
                data=json.dumps(stats_report, indent=2),
                file_name=f"Statistics_Report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

else:  # About page
    st.markdown("## ℹ️ About the System")
    
    st.markdown("""
    <div class='info-card'>
        <h2 style='color: #1F3C88;'>🧬 Blastocyst Grading System v2.0</h2>
        <p style='font-size: 16px; line-height: 1.8;'>
            Advanced AI-powered embryo quality assessment platform designed for 
            assisted reproductive technology (ART) clinics and IVF laboratories.
        </p>
        
        <h3 style='color: #1F3C88; margin-top: 25px;'>✨ Key Features</h3>
        <ul style='font-size: 15px; line-height: 2;'>
            <li>🔬 Automated grading of ICM, TE, and Expansion parameters</li>
            <li>🎯 Success probability estimation based on embryo quality</li>
            <li>📄 Professional PDF report generation</li>
            <li>📜 Comprehensive analysis history tracking</li>
            <li>📊 Statistical analysis and quality trends</li>
            <li>💾 Multiple export formats (PDF, JSON, CSV)</li>
            <li>👨‍⚕️ Detailed clinical recommendations</li>
            <li>🔍 Advanced search and filtering capabilities</li>
        </ul>
        
        <h3 style='color: #1F3C88; margin-top: 25px;'>🎓 Grading Methodology</h3>
        <p style='font-size: 15px; line-height: 1.8;'>
            The system evaluates three critical parameters using a 1-5 grading scale:
        </p>
        <ul style='font-size: 15px; line-height: 2;'>
            <li><b>ICM (Inner Cell Mass):</b> Evaluates the quality and compactness of cells 
            that will develop into the fetus</li>
            <li><b>TE (Trophectoderm):</b> Assesses the quality of cells that will form 
            the placenta and support structures</li>
            <li><b>EXP (Expansion):</b> Measures the developmental stage and expansion 
            of the blastocyst cavity</li>
        </ul>
        
        <h3 style='color: #1F3C88; margin-top: 25px;'>⚠️ Important Disclaimer</h3>
        <p style='font-size: 14px; line-height: 1.8; color: #d32f2f;'>
            <b>This system is designed as a clinical decision support tool.</b> All results 
            must be reviewed and validated by qualified embryologists and reproductive 
            medicine specialists. Clinical decisions should be made based on comprehensive 
            patient evaluation, complete medical history, and professional medical judgment. 
            The success probabilities are statistical estimates and may not reflect 
            individual patient outcomes.
        </p>
        
        <h3 style='color: #1F3C88; margin-top: 25px;'>📞 Support & Contact</h3>
        <p style='font-size: 15px;'>
            For technical support, training, or integration inquiries:<br/>
            📧 Email: support@embryograding.com<br/>
            📱 Phone: +1 (555) 123-4567<br/>
            🌐 Website: www.embryograding.com
        </p>
        
        <p style='margin-top: 30px; text-align: center; color: #666; font-size: 13px;'>
            Version 2.0 | Last Updated: January 2026<br/>
            © 2026 Blastocyst Grading System. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grading Reference Tables
    st.markdown("### 📚 Detailed Grading Reference")
    
    tab1, tab2, tab3 = st.tabs(["ICM Grading", "TE Grading", "EXP Grading"])
    
    with tab1:
        st.markdown("""
        | Grade | Description | Clinical Significance |
        |-------|-------------|----------------------|
        | 5 | Many cells, tightly packed | Excellent quality, optimal for transfer |
        | 4 | Several cells, loosely grouped | Very good quality, high success rate |
        | 3 | Few cells | Good quality, acceptable for transfer |
        | 2 | Very few cells | Fair quality, reduced viability |
        | 1 | Poor, degenerating cells | Poor quality, low success probability |
        """)
    
    with tab2:
        st.markdown("""
        | Grade | Description | Clinical Significance |
        |-------|-------------|----------------------|
        | 5 | Many cells, cohesive epithelium | Excellent implantation potential |
        | 4 | Several cells, loose epithelium | Very good implantation potential |
        | 3 | Few cells | Good implantation potential |
        | 2 | Very few cells | Fair implantation potential |
        | 1 | Poor, few large cells | Poor implantation potential |
        """)
    
    with tab3:
        st.markdown("""
        | Grade | Description | Clinical Significance |
        |-------|-------------|----------------------|
        | 5 | Hatching/hatched blastocyst | Advanced development, ready for transfer |
        | 4 | Expanded, thin zona pellucida | Well-developed, optimal timing |
        | 3 | Full blastocyst, thick zona | Standard development |
        | 2 | Early blastocyst, small cavity | Early development stage |
        | 1 | Barely visible cavity | Very early development |
        """)


# Footer with logo
def add_logo():
    try:
        with open("logo.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='position: fixed; right: 25px; bottom: 25px; z-index: 9999;'>
                <img src='data:image/png;base64,{encoded}' 
                     style='width: 180px; opacity: 0.92; border-radius: 10px; 
                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

add_logo()