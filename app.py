import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


#------------Inferences------------
def get_inference(icm, te, exp):
    notes = []

    # ICM (Inner Cell Mass)
    if icm >= 4:
        notes.append(
            "🟢 **ICM:** Good inner cell mass – likely healthy embryoblast."
        )
    elif icm == 3:
        notes.append(
            "🟡 **ICM:** Moderate ICM – acceptable but not ideal."
        )
    else:
        notes.append(
            "🔴 **ICM:** Poor ICM – lower likelihood of successful implantation."
        )

    # TE (Trophectoderm)
    if te >= 4:
        notes.append(
            "🟢 **TE:** Strong trophectoderm – better implantation probability."
        )
    elif te == 3:
        notes.append(
            "🟡 **TE:** Average TE – may still be viable."
        )
    else:
        notes.append(
            "🔴 **TE:** Weak TE – may reduce implantation chances."
        )

    # EXP (Expansion grade)
    if exp >= 4:
        notes.append(
            "🟢 **Expansion:** Well-expanded blastocyst – good development."
        )
    elif exp == 3:
        notes.append(
            "🟡 **Expansion:** Moderate expansion – monitor carefully."
        )
    else:
        notes.append(
            "🔴 **Expansion:** Poor expansion – embryo may be underdeveloped."
        )

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
    
    # Calculate overall quality score
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
    
    # Specific recommendations based on individual parameters
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


def generate_detailed_report(icm, te, exp, img_array, patient_info=None):
    """Generate a detailed medical report"""
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report.append("=" * 80)
    report.append("BLASTOCYST GRADING REPORT".center(80))
    report.append("=" * 80)
    report.append(f"\nReport Generated: {timestamp}")
    report.append(f"Analysis System: AI-Powered Embryo Grading System v1.0")
    
    if patient_info:
        report.append("\n" + "-" * 80)
        report.append("PATIENT INFORMATION")
        report.append("-" * 80)
        for key, value in patient_info.items():
            report.append(f"{key}: {value}")
    
    report.append("\n" + "-" * 80)
    report.append("GRADING RESULTS")
    report.append("-" * 80)
    report.append(f"Inner Cell Mass (ICM): {icm}/5")
    report.append(f"Trophectoderm (TE): {te}/5")
    report.append(f"Expansion Grade (EXP): {exp}/5")
    
    # Calculate quality score
    avg_score = (icm + te + exp) / 3
    report.append(f"\nAverage Quality Score: {avg_score:.2f}/5")
    
    if avg_score >= 4:
        quality = "EXCELLENT"
    elif avg_score >= 3:
        quality = "MODERATE"
    else:
        quality = "POOR"
    report.append(f"Overall Quality Assessment: {quality}")
    
    report.append("\n" + "-" * 80)
    report.append("DETAILED ANALYSIS")
    report.append("-" * 80)
    
    summary, notes = get_inference(icm, te, exp)
    report.append(f"\n{summary}")
    for note in notes:
        # Remove emoji and markdown formatting for text report
        clean_note = note.replace("🟢", "[GOOD]").replace("🟡", "[MODERATE]").replace("🔴", "[POOR]")
        clean_note = clean_note.replace("**", "")
        report.append(clean_note)
    
    report.append("\n" + "-" * 80)
    report.append("CLINICAL RECOMMENDATIONS")
    report.append("-" * 80)
    
    suggestions = get_doctor_suggestions(icm, te, exp)
    for suggestion in suggestions:
        clean_suggestion = suggestion.replace("**", "")
        report.append(clean_suggestion)
    
    report.append("\n" + "-" * 80)
    report.append("GRADING CRITERIA REFERENCE")
    report.append("-" * 80)
    report.append("\nInner Cell Mass (ICM) Grading:")
    report.append("  Grade 5: Many cells, tightly packed")
    report.append("  Grade 4: Several cells, loosely grouped")
    report.append("  Grade 3: Few cells")
    report.append("  Grade 2: Very few cells")
    report.append("  Grade 1: Poor, degenerating cells")
    
    report.append("\nTrophectoderm (TE) Grading:")
    report.append("  Grade 5: Many cells forming cohesive epithelium")
    report.append("  Grade 4: Several cells forming loose epithelium")
    report.append("  Grade 3: Few cells")
    report.append("  Grade 2: Very few cells")
    report.append("  Grade 1: Poor, few large cells")
    
    report.append("\nExpansion (EXP) Grading:")
    report.append("  Grade 5: Hatching/hatched blastocyst")
    report.append("  Grade 4: Expanded, thin zona pellucida")
    report.append("  Grade 3: Full blastocyst, thick zona")
    report.append("  Grade 2: Early blastocyst, small cavity")
    report.append("  Grade 1: Early blastocyst, barely visible cavity")
    
    report.append("\n" + "=" * 80)
    report.append("DISCLAIMER")
    report.append("=" * 80)
    report.append("This report is generated by an AI-assisted analysis system and should be")
    report.append("reviewed and validated by qualified embryologists and fertility specialists.")
    report.append("Clinical decisions should be made in conjunction with comprehensive patient")
    report.append("evaluation and professional medical judgment.")
    report.append("=" * 80)
    
    return "\n".join(report)


IMG_SIZE = 224

st.markdown(
    """
    <style>
    .center-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
    }

    .center-box h3, 
    .center-box p, 
    .center-box div {
        color: #6e6e6e;
        font-weight: 600;
    }
    
    .suggestion-box {
        background: rgba(255, 248, 220, 0.95);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FFA500;
        margin: 15px 0;
        text-align: left;
    }
    
    .download-section {
        margin-top: 25px;
        padding: 20px;
        background: rgba(230, 240, 255, 0.9);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- BACKGROUND ----------------
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
        pass  # If background image not found, continue without it


add_bg("background1.jpg")


# ---------------- MODEL ----------------
model = tf.keras.models.load_model("embryo_output_model.keras")


# ---------------- CENTER LAYOUT ----------------
st.markdown("""
<style>
.center-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='center-container'>", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#1F3C88;
    text-shadow:0px 2px 6px rgba(0,0,0,0.25);'>
        🧬 Blastocyst Grading
    </h1>
    """,
    unsafe_allow_html=True
)

# Optional patient information
with st.expander("📋 Add Patient Information (Optional)"):
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID")
        patient_age = st.number_input("Age", min_value=18, max_value=50, value=30)
    with col2:
        cycle_day = st.text_input("Cycle Day", value="Day 5")
        embryo_id = st.text_input("Embryo ID")

uploaded = st.file_uploader("Upload embryo image", type=["png", "jpg", "jpeg"])


# ---------------- PREDICTION ----------------
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


# ---------------- SHOW OUTPUT ----------------
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded embryo", use_container_width=True)

    icm, te, exp = predict(img)

    # Prepare patient info dictionary
    patient_info = {}
    if patient_id:
        patient_info["Patient ID"] = patient_id
    if patient_age:
        patient_info["Patient Age"] = str(patient_age)
    if cycle_day:
        patient_info["Cycle Day"] = cycle_day
    if embryo_id:
        patient_info["Embryo ID"] = embryo_id

    # prediction card styling
    st.markdown(
        """
        <style>

        .prediction-card {
            background: rgba(255,255,255,0.82);
            padding: 20px;
            border-radius: 16px;
            max-width: 520px;
            margin-left: auto;
            margin-right: auto;
            margin-top: 10px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.12);
        }

        .prediction-card * {
            color: #0F265C !important;
            font-weight: 700;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="center-box">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            text-align: center;
            color: #000000;
            font-size: 26px;
            font-weight: 800;
            margin-top: 10px;
        ">
            Prediction
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            text-align: center;
            color: #000000;
            font-size: 18px;
            font-weight: 600;
            margin-top: 6px;
        ">
            ICM: {icm}
            <br>
            TE : {te}
            <br>
            EXP: {exp}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    summary, notes = get_inference(icm, te, exp)

    st.markdown(
        f"""
        <div style="
            text-align: center;
            color: #000000;
            font-size: 24px;
            font-weight: 800;
            margin-top: 10px;
        ">
            Inference
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            text-align: center;
            color: #000000;
            font-size: 18px;
            font-weight: 600;
            margin-top: 10px;
        ">
            {summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    for n in notes:
        st.markdown(
            f"""
            <div style="
                text-align: center;
                color: #000000;
                font-size: 16px;
                margin-top: 4px;
            ">
                {n}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Doctor Suggestions Section
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="
            text-align: center;
            color: #1F3C88;
            font-size: 24px;
            font-weight: 800;
            margin-top: 20px;
        ">
            👨‍⚕️ Clinical Recommendations
        </div>
        """,
        unsafe_allow_html=True
    )
    
    suggestions = get_doctor_suggestions(icm, te, exp)
    
    suggestions_html = "<div class='suggestion-box' style='text-align: left;'>"
    for suggestion in suggestions:
        suggestions_html += f"<p style='margin: 8px 0; color: #333;'>{suggestion}</p>"
    suggestions_html += "</div>"
    
    st.markdown(suggestions_html, unsafe_allow_html=True)
    
    # Download Report Section
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="
            text-align: center;
            color: #1F3C88;
            font-size: 24px;
            font-weight: 800;
            margin-top: 20px;
        ">
            📥 Download Report
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Generate report
    report_text = generate_detailed_report(icm, te, exp, img, patient_info if patient_info else None)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Download as text file
        st.download_button(
            label="📄 Download Text Report",
            data=report_text,
            file_name=f"blastocyst_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Preview report in expander
        with st.expander("👁️ Preview Full Report"):
            st.text(report_text)

# close center block
st.markdown("</div>", unsafe_allow_html=True)

def add_logo():
    try:
        with open("logo.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .logo-container {{
                position: fixed;
                right: 25px;
                bottom: 25px;
                z-index: 9999;
            }}

            .logo-container img {{
                width: 180px;
                opacity: 0.92;
            }}
            </style>

            <div class="logo-container">
                <img src="data:image/png;base64,{encoded}">
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        pass  # If logo not found, continue without it

add_logo()