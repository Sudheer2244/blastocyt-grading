import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import base64


#------------Inferences------------
def get_inference(icm, te, exp):
    notes = []

    # ICM (Inner Cell Mass)
    if icm >= 4:
        notes.append(
            "🟢 **ICM:** Good inner cell mass — likely healthy embryoblast."
        )
    elif icm == 3:
        notes.append(
            "🟡 **ICM:** Moderate ICM — acceptable but not ideal."
        )
    else:
        notes.append(
            "🔴 **ICM:** Poor ICM — lower likelihood of successful implantation."
        )

    # TE (Trophectoderm)
    if te >= 4:
        notes.append(
            "🟢 **TE:** Strong trophectoderm — better implantation probability."
        )
    elif te == 3:
        notes.append(
            "🟡 **TE:** Average TE — may still be viable."
        )
    else:
        notes.append(
            "🔴 **TE:** Weak TE — may reduce implantation chances."
        )

    # EXP (Expansion grade)
    if exp >= 4:
        notes.append(
            "🟢 **Expansion:** Well-expanded blastocyst — good development."
        )
    elif exp == 3:
        notes.append(
            "🟡 **Expansion:** Moderate expansion — monitor carefully."
        )
    else:
        notes.append(
            "🔴 **Expansion:** Poor expansion — embryo may be underdeveloped."
        )

    # Overall summary
    if icm >= 4 and te >= 4 and exp >= 4:
        summary = "✅ **Overall:** High-quality blastocyst — strong transfer candidate."
    elif icm >= 3 and te >= 3 and exp >= 3:
        summary = "⚠️ **Overall:** Medium-quality embryo — possible, but requires clinical judgement."
    else:
        summary = "❌ **Overall:** Low-quality blastocyst — poor prognosis."

    return summary, notes



IMG_SIZE = 224
#Prediction BG and align
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
        color: #6e6e6e;   /* light grey */
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ---------------- BACKGROUND ----------------
def add_bg(image_file):
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


add_bg("C:\\Users\\sudhe\\OneDrive\\Desktop\\Blastocyst grading\\background1.jpg")


# ---------------- MODEL ----------------
model = tf.keras.models.load_model("blastocyst_exp_focused.keras")


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

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#3a3a3a;
            font-size:14px;
            margin-top:14px;
            font-style:italic;
        ">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# close center block
st.markdown("</div>", unsafe_allow_html=True)

def add_logo():
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

add_logo()

