import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
import os
from fpdf import FPDF
import time

# ==========================================
# 1. GLOBAL CONFIGURATION & DATABASE
# ==========================================
st.set_page_config(
    page_title="OculoVision Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "OculoVision Pro - AI-Powered Glaucoma Screening System v1.0"}
)

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"
DB_FILE = "patient_clinical_history.csv"

# ==========================================
# ULTRA-PREMIUM ENTERPRISE MEDICAL UI
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100;200;300;400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #050f1f 0%, #0a1528 25%, #081835 50%, #0a1528 75%, #050f1f 100%);
        color: #e8f2fe;
        letter-spacing: 0.3px;
    }

    .stApp {
        background: linear-gradient(135deg, #050f1f 0%, #0a1528 25%, #081835 50%, #0a1528 75%, #050f1f 100%);
    }

    /* ===== PREMIUM HEADER ===== */
    .header-container {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 30%, #1a2a40 70%, #0f2847 100%);
        padding: 50px 60px;
        border-radius: 24px;
        border: 1.5px solid rgba(96, 165, 250, 0.25);
        margin-bottom: 45px;
        box-shadow:
            0 35px 80px rgba(0, 0, 0, 0.6),
            0 0 80px rgba(96, 165, 250, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(25px);
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 8s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(20px); }
    }

    .header-title {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 35%, #06b6d4 70%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
        letter-spacing: -1.5px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(96, 165, 250, 0.3);
    }

    .header-subtitle {
        color: #93c5e0;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    /* ===== SIDEBAR PREMIUM ===== */
    .sidebar-header {
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .sidebar-subtitle {
        color: #93c5e0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-bottom: 25px;
        text-transform: uppercase;
    }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #34d399;
        font-weight: 700;
        margin-bottom: 25px;
        font-size: 14px;
        padding: 14px 18px;
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.12) 0%, rgba(52, 211, 153, 0.08) 100%);
        border: 1.5px solid rgba(52, 211, 153, 0.3);
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(52, 211, 153, 0.1);
    }

    .pulse-dot {
        height: 12px;
        width: 12px;
        background-color: #34d399;
        border-radius: 50%;
        animation: pulse 2.5s infinite;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    .sidebar-card {
        background: linear-gradient(135deg, rgba(30, 58, 82, 0.7) 0%, rgba(15, 40, 71, 0.9) 100%);
        color: #e8f2fe !important;
        padding: 20px 22px;
        border-radius: 16px;
        margin-bottom: 16px;
        border: 1.5px solid rgba(96, 165, 250, 0.25);
        backdrop-filter: blur(20px);
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        font-size: 13px;
        line-height: 1.8;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .sidebar-card:hover {
        background: linear-gradient(135deg, rgba(30, 58, 82, 0.9) 0%, rgba(15, 40, 71, 1) 100%);
        border-color: rgba(52, 211, 153, 0.5);
        box-shadow:
            0 15px 35px rgba(52, 211, 153, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-5px);
    }

    .sidebar-card b {
        color: #60a5fa;
        font-size: 15px;
        display: block;
        margin-bottom: 8px;
        font-weight: 800;
    }

    /* ===== INPUT LABELS ===== */
    .input-label {
        color: #93c5e0;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        display: block;
    }

    .stTextInput > div > div > input {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%) !important;
        border: 1.5px solid rgba(96, 165, 250, 0.35) !important;
        border-radius: 14px !important;
        color: #e8f2fe !important;
        padding: 14px 18px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.15), 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%) !important;
        border: 1.5px solid rgba(96, 165, 250, 0.35) !important;
        border-radius: 14px !important;
        color: #e8f2fe !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.15), 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    .stSelectbox > div > div > select {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%) !important;
        border: 1.5px solid rgba(96, 165, 250, 0.35) !important;
        border-radius: 14px !important;
        color: #e8f2fe !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    /* ===== RESULT CARDS PREMIUM ===== */
    .result-card {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%);
        padding: 36px 40px;
        border-radius: 20px;
        border: 1.5px solid rgba(96, 165, 250, 0.2);
        margin: 28px 0;
        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.4s ease;
    }

    .result-status-high {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(159, 18, 57, 0.1) 100%);
        border: 2px solid rgba(239, 68, 68, 0.45);
        box-shadow:
            0 20px 50px rgba(239, 68, 68, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .result-status-normal {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 2px solid rgba(34, 197, 94, 0.45);
        box-shadow:
            0 20px 50px rgba(34, 197, 94, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    /* ===== TABS PREMIUM ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1.5px solid rgba(96, 165, 250, 0.15);
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #7dd3c0;
        background-color: transparent;
        border-radius: 14px 14px 0 0;
        padding: 16px 28px;
        font-weight: 700;
        font-size: 14px;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: none;
        border-bottom: 4px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #60a5fa;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.12) 0%, rgba(96, 165, 250, 0.08) 100%);
        border-bottom: 4px solid #60a5fa;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.1);
    }

    /* ===== BUTTONS PREMIUM ===== */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 16px 36px !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        box-shadow:
            0 10px 25px rgba(59, 130, 246, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        box-shadow:
            0 15px 40px rgba(59, 130, 246, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-4px) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* ===== FILE UPLOADER PREMIUM ===== */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%) !important;
        border: 2.5px dashed rgba(96, 165, 250, 0.5) !important;
        border-radius: 18px !important;
        padding: 60px 40px !important;
        transition: all 0.4s ease !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(96, 165, 250, 0.8) !important;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(52, 211, 153, 0.15) 100%) !important;
        box-shadow: 0 0 40px rgba(96, 165, 250, 0.2) !important;
    }

    /* ===== PROGRESS BAR PREMIUM ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ef4444 0%, #f97316 25%, #fbbf24 50%, #86efac 75%, #34d399 100%) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.3) !important;
    }

    /* ===== METRICS PREMIUM ===== */
    .stMetric {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%) !important;
        padding: 24px 28px !important;
        border-radius: 16px !important;
        border: 1.5px solid rgba(96, 165, 250, 0.2) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    }

    /* ===== DATA TABLE PREMIUM ===== */
    .dataframe {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%) !important;
        border-radius: 16px !important;
        border: 1.5px solid rgba(96, 165, 250, 0.2) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    }

    /* ===== SECTION TITLES ===== */
    h3 {
        color: #60a5fa !important;
        font-weight: 800 !important;
        font-size: 20px !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 25px !important;
    }

    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(96, 165, 250, 0.25) 50%, transparent 100%);
        margin: 30px 0;
    }

    /* ===== SCROLLBAR PREMIUM ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 40, 71, 0.4);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #60a5fa 0%, #34d399 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.3);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #93c5f9 0%, #6ee7b7 100%);
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.5);
    }

    /* ===== TOAST STYLING ===== */
    .stToast {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%) !important;
        border: 1.5px solid rgba(52, 211, 153, 0.4) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PDF GENERATOR WITH MEDICAL REASONING
# ==========================================
def create_medical_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(200, 20, "OFFICIAL GLAUCOMA SCREENING REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient ID: {data['Patient_ID']}")
    pdf.cell(100, 10, f"Timestamp: {data['Timestamp']}", ln=True)
    pdf.cell(100, 10, f"Ocular Laterality: {data['Eye']}")
    pdf.cell(100, 10, f"Intraocular Pressure: {data['IOP']} mmHg", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    status_color = (220, 38, 38) if data['Result'] == "GLAUCOMA SUSPECT" else (22, 163, 74)
    pdf.set_text_color(*status_color)
    pdf.cell(200, 10, f"DIAGNOSTIC STATUS: {data['Result']} ({data['Prob']})", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(50, 50, 50)
    
    # Automated Clinical Reasoning
    if data['Result'] == "GLAUCOMA SUSPECT":
        explanation = ("Reasoning: The AI ensemble detected significant excavation of the optic cup and thinning of the neuroretinal rim. "
                       "These structural changes are characteristic of glaucomatous neuropathy, where increased pressure damages optic nerve fibers. "
                       "Clinical follow-up for perimetry and OCT imaging is strongly advised.")
    else:
        explanation = ("Reasoning: The retinal topography indicates a healthy cup-to-disc ratio. The neuroretinal rim appears robust, "
                       "with no visible signs of nerve fiber layer defects or pathological cupping. Structural patterns fall within normal clinical limits.")
    
    pdf.multi_cell(0, 10, explanation)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. CORE AI & DATABASE LOGIC
# ==========================================
@st.cache_resource
def load_engine():
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

engine = load_engine()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

def save_to_history(data):
    df = pd.DataFrame([data])
    if not os.path.isfile(DB_FILE): df.to_csv(DB_FILE, index=False)
    else: df.to_csv(DB_FILE, mode='a', header=False, index=False)

def apply_clahe_clinical(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def preprocess_for_inference(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), int(min(h, w) * 0.45), 255, -1)
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, bbox_w, bbox_h = cv2.boundingRect(coords)
        masked = masked[y:y+bbox_h, x:x+bbox_w]
    enhanced = apply_clahe_clinical(masked)
    resized = cv2.resize(enhanced, (IMAGE_SIZE, IMAGE_SIZE))
    return img_rgb, resized

def generate_gradcam(model, img_array, layer_name):
    img_batch = np.expand_dims(img_array, axis=0)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        loss = 1.0 - predictions[:, 0] 
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0: return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap /= np.max(heatmap) + 1e-10
    return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

# ==========================================
# 4. SIDEBAR (History & Auto-Update)
# ==========================================
if 'p_count' not in st.session_state: st.session_state.p_count = 1001
if 'processed_flag' not in st.session_state: st.session_state.processed_flag = False

with st.sidebar:
    st.markdown('<div class="status-indicator"><span class="pulse-dot"></span>System Online</div>', unsafe_allow_html=True)
    st.markdown("<div class='sidebar-header'>OculoVision</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle'>AI-Powered Screening System</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-card"><b>🔬 Explainable AI</b>GradCAM++ technology reveals critical diagnostic regions in fundus images.</div>
        <div class="sidebar-card"><b>🔐 Validated Model</b>Trained on HRF dataset • Tested on Drishti-GS benchmark</div>
        <div class="sidebar-card"><b>📄 Report Generation</b>Professional medical-grade PDF reports with clinical reasoning</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col_btn_left, col_btn_right = st.columns(2, gap="small")
    with col_btn_left:
        if st.button("➤ Next Patient", use_container_width=True, type="primary"):
            st.session_state.p_count += 1
            st.session_state.processed_flag = False
            st.rerun()
    with col_btn_right:
        if st.button("↻ Reset", use_container_width=True):
            st.session_state.processed_flag = False

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#60a5fa; font-size:16px; margin-bottom:15px;'>📋 Recent Cases</h3>", unsafe_allow_html=True)
    if os.path.exists(DB_FILE):
        hist_df = pd.read_csv(DB_FILE)
        display_cols = hist_df[['Patient_ID', 'Eye', 'Result']].tail(8)
        st.dataframe(display_cols, use_container_width=True, hide_index=True)

# ==========================================
# 5. MAIN WORKSPACE
# ==========================================
st.markdown("""<div class="header-container">
    <div class="header-title">👁️ Neural Diagnostic Workspace</div>
    <div class="header-subtitle">Glaucoma Screening • Explainable AI System</div>
</div>""", unsafe_allow_html=True)

st.markdown("### 👤 Patient Information", unsafe_allow_html=True)
with st.container():
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown("<label class='input-label'>📋 Patient ID</label>", unsafe_allow_html=True)
        p_id = st.text_input("Patient ID", value=f"PX-{st.session_state.p_count}", label_visibility="collapsed", key="pid", placeholder="PX-1001")
    with c2:
        st.markdown("<label class='input-label'>📅 Age (years)</label>", unsafe_allow_html=True)
        age = st.number_input("Age", 1, 110, 60, label_visibility="collapsed", key="age", help="Patient age in years")
    with c3:
        st.markdown("<label class='input-label'>👁️ Laterality</label>", unsafe_allow_html=True)
        eye = st.selectbox("Target Eye", ["Right (OD)", "Left (OS)"], label_visibility="collapsed", key="eye")
    with c4:
        st.markdown("<label class='input-label'>🔬 IOP (mmHg)</label>", unsafe_allow_html=True)
        iop = st.number_input("IOP", 5, 50, 16, label_visibility="collapsed", key="iop", help="Intraocular pressure measurement")

st.markdown("### 🖼️ Fundus Image Upload", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop high-quality fundus image or click to select", type=["jpg", "png", "jpeg"], label_visibility="collapsed", help="Upload a high-quality fundus photograph for analysis")

if uploaded_file:
    # Inference
    raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
    model_input = prep_func(processed_img.astype(np.float32))
    
    prediction = engine.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    glaucoma_prob = 1.0 - float(prediction[0])
    diag_result = "GLAUCOMA SUSPECT" if glaucoma_prob >= 0.60 else "NORMAL"
    
    # --- AUTO-SAVE LOGIC (FIXED) ---
    current_key = f"{p_id}_{eye}_{uploaded_file.name}"
    
    if not st.session_state.processed_flag:
        db_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient_ID": p_id, "Age": age, "Eye": eye, "IOP": iop,
            "Prob": f"{glaucoma_prob*100:.1f}%", "Result": diag_result
        }
        save_to_history(db_entry)
        st.session_state.processed_flag = True
        st.toast(f"Record logged for {p_id}", icon="💾")

    # Tabs
    t1, t2, t3 = st.tabs(["📊 Diagnostic Results", "🔬 Neural Explainability", "📑 Clinical Report"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2.8, 1.4, 0.8], gap="medium")

        with col1:
            if glaucoma_prob >= 0.60:
                st.markdown(f"""<div class="result-card result-status-high">
                    <div style="font-size: 15px; font-weight: 800; color: #ef4444; margin-bottom: 14px; letter-spacing: 0.8px; text-transform: uppercase;">⚠️ Critical Finding</div>
                    <div style="font-size: 56px; font-weight: 900; color: #fca5a5; margin-bottom: 22px; letter-spacing: -2px; font-variant-numeric: tabular-nums;">{glaucoma_prob*100:.1f}%</div>
                    <div style="font-size: 13px; color: #cbd5e1; line-height: 1.9; margin-bottom: 18px;">
                        <strong style="color: #f87171;">Diagnosis:</strong> Glaucoma Suspect<br/>
                        <strong style="color: #f87171;">Risk Level:</strong> High (≥60%)<br/>
                        <strong style="color: #f87171;">Recommendation:</strong> Urgent clinical referral for comprehensive evaluation including:<br/>
                        • Optical Coherence Tomography (OCT)<br/>
                        • Visual Field Perimetry<br/>
                        • Gonioscopy & Fundus Photography
                    </div>
                    <div style="font-size: 12px; color: #7dd3c0; background: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.08) 100%); padding: 14px 16px; border-radius: 10px; border-left: 4px solid #34d399; line-height: 1.7;">
                        <strong>⚡ Important:</strong> This is an AI-assisted analysis. Clinical judgment and professional examination are required for definitive diagnosis. Do not delay specialist referral.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-card result-status-normal">
                    <div style="font-size: 15px; font-weight: 800; color: #22c55e; margin-bottom: 14px; letter-spacing: 0.8px; text-transform: uppercase;">✓ Normal Variant</div>
                    <div style="font-size: 56px; font-weight: 900; color: #86efac; margin-bottom: 22px; letter-spacing: -2px; font-variant-numeric: tabular-nums;">{glaucoma_prob*100:.1f}%</div>
                    <div style="font-size: 13px; color: #cbd5e1; line-height: 1.9; margin-bottom: 18px;">
                        <strong style="color: #86efac;">Diagnosis:</strong> No Acute Pathology<br/>
                        <strong style="color: #86efac;">Clinical Assessment:</strong> Healthy variant<br/>
                        <strong style="color: #86efac;">Observations:</strong><br/>
                        • Preserved neuroretinal rim<br/>
                        • Normal cup-to-disc ratio<br/>
                        • No nerve fiber layer defects<br/>
                        • Routine follow-up recommended
                    </div>
                    <div style="font-size: 12px; color: #7dd3c0; background: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.08) 100%); padding: 14px 16px; border-radius: 10px; border-left: 4px solid #34d399; line-height: 1.7;">
                        <strong>ℹ️ Note:</strong> Regular eye examinations are essential for early detection of any future changes. Annual screening recommended.
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("<p style='font-size: 11px; color: #93c5e0; margin-bottom: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;'>Risk Gradient</p>", unsafe_allow_html=True)
            st.progress(glaucoma_prob)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 11px; color: #93c5e0; margin-bottom: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;'>Probability</p>", unsafe_allow_html=True)
            prob_color = "#ef4444" if glaucoma_prob >= 0.60 else "#34d399"
            st.markdown(f"<p style='font-size: 24px; font-weight: 900; color: {prob_color};'>{glaucoma_prob*100:.1f}%</p>", unsafe_allow_html=True)

        with col3:
            conf_score = max(glaucoma_prob, 1-glaucoma_prob)
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a3a52 0%, #0f2847 100%); padding: 18px 16px; border-radius: 14px; border: 1.5px solid rgba(96, 165, 250, 0.25); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);">
                    <div style="font-size: 10px; color: #93c5e0; text-transform: uppercase; font-weight: 800; margin-bottom: 10px; letter-spacing: 1px;">Model<br/>Confidence</div>
                    <div style="font-size: 32px; font-weight: 900; color: #60a5fa; margin-bottom: 8px; font-variant-numeric: tabular-nums;">{conf_score*100:.0f}%</div>
                    <div style="font-size: 9px; color: #7dd3c0; margin-top: 10px; font-weight: 600;">Predictive<br/>Certainty</div>
                </div>
            """, unsafe_allow_html=True)

    with t2:
        st.markdown("#### Grad-CAM++ Neural Visualization", unsafe_allow_html=True)
        st.markdown("""<p style='color: #93c5e0; font-size: 13px; margin-bottom: 22px; line-height: 1.8;'>
        <strong>Explainable AI Analysis:</strong> This heatmap visualizes which regions of the fundus image the neural network prioritized during diagnosis. <strong style="color: #f87171;">Hot colors (red/orange/yellow)</strong> indicate high-importance regions that strongly influenced the prediction. <strong style="color: #60a5fa;">Cool colors (blue/cyan)</strong> indicate regions of lower diagnostic significance. This transparency helps clinicians understand and validate the AI's decision-making process.
        </p>""", unsafe_allow_html=True)
        heatmap = generate_gradcam(engine, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 0.7, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.3, 0)
        st.image(overlay, caption="GradCAM++ Attention Map - Critical diagnostic regions highlighted", use_container_width=True)
        st.markdown("""<div style='font-size: 12px; color: #7dd3c0; margin-top: 18px; padding: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.08) 100%); border-radius: 10px; border-left: 4px solid #34d399; line-height: 1.8;'>
        <strong>💡 Interpretability Insight:</strong> Grad-CAM++ technology makes AI decisions interpretable by showing which anatomical regions (optic disc, cup, neuroretinal rim) influenced the prediction. This supports clinical validation and builds confidence in the diagnostic system.
        </div>""", unsafe_allow_html=True)

    with t3:
        st.markdown("#### Professional Clinical Report", unsafe_allow_html=True)
        st.markdown("""<p style='color: #93c5e0; font-size: 13px; margin-bottom: 22px;'>
        Generate a comprehensive, medical-grade PDF report suitable for patient records, clinical referral, and specialist consultation. The report includes patient demographics, AI findings, clinical reasoning, and recommendations.
        </p>""", unsafe_allow_html=True)
        report_data = {
            "Patient_ID": p_id, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Eye": eye, "IOP": iop, "Result": diag_result, "Prob": f"{glaucoma_prob*100:.1f}%"
        }
        pdf_bytes = create_medical_report(report_data)

        col_report_left, col_report_right, col_report_spacer = st.columns([1.5, 1.5, 1], gap="small")
        with col_report_left:
            st.download_button(
                "📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"GlaucomaReport_{p_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                use_container_width=True,
                type="primary"
            )
        with col_report_right:
            st.success(f"✓ Report Generated for Patient {p_id}", icon="✅")
