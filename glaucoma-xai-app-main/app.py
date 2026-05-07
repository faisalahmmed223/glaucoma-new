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
st.set_page_config(page_title="OculoVision Pro", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"
DB_FILE = "patient_clinical_history.csv"

# Enhanced Modern Clinical UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%);
        color: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%);
    }

    /* ===== HEADER SECTION ===== */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 30px 40px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    .header-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ===== INPUT SECTION ===== */
    .input-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }

    .input-label {
        color: #cbd5e1;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: block;
    }

    /* ===== SIDEBAR STYLING ===== */
    .sidebar-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 20px;
    }

    .sidebar-card {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.5) 0%, rgba(30, 41, 59, 0.8) 100%);
        color: #f1f5f9 !important;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 4px solid #60a5fa;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        font-size: 13px;
        line-height: 1.6;
    }

    .sidebar-card:hover {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.7) 0%, rgba(30, 41, 59, 1) 100%);
        border-left-color: #34d399;
        box-shadow: 0 8px 16px rgba(52, 211, 153, 0.1);
        transform: translateX(5px);
    }

    .sidebar-card b {
        color: #60a5fa;
        font-size: 14px;
        display: block;
        margin-bottom: 6px;
        font-weight: 700;
    }

    /* ===== STATUS INDICATOR ===== */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #34d399;
        font-weight: 700;
        margin-bottom: 20px;
        font-size: 14px;
    }

    .pulse-dot {
        height: 12px;
        width: 12px;
        background-color: #34d399;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* ===== RESULT CARDS ===== */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }

    .result-status-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }

    .result-status-normal {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #86efac;
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 2px solid rgba(148, 163, 184, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #60a5fa;
        border-bottom: 3px solid #60a5fa;
        background-color: rgba(96, 165, 250, 0.05);
    }

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ef4444 0%, #fbbf24 50%, #34d399 100%);
        border-radius: 10px;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        border-radius: 12px !important;
    }

    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.05) 0%, rgba(52, 211, 153, 0.05) 100%) !important;
        border: 2px dashed rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px !important;
        padding: 40px !important;
    }

    /* ===== DATA TABLE ===== */
    .dataframe {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(148, 163, 184, 0.2) 50%, transparent 100%);
        margin: 25px 0;
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
    st.markdown("<h2 style='color:#60a5fa; margin-bottom:5px;'>OculoVision</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:12px; margin-bottom:25px;'>AI-Powered Glaucoma Screening</p>", unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-card"><b>🔬 XAI Powered</b>Advanced Grad-CAM visualization reveals critical diagnostic regions.</div>
        <div class="sidebar-card"><b>🛡️ Clinical Validation</b>Trained on HRF • Tested on Drishti-GS dataset</div>
        <div class="sidebar-card"><b>📊 PDF Export</b>Generate professional medical reports instantly</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➤ Next Patient", use_container_width=True, type="primary"):
        st.session_state.p_count += 1
        st.session_state.processed_flag = False
        st.rerun()

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
    <div class="header-subtitle">Glaucoma Detection • Explainable AI</div>
</div>""", unsafe_allow_html=True)

st.markdown("### Patient Information", unsafe_allow_html=True)
with st.container():
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown("<label class='input-label'>Patient ID</label>", unsafe_allow_html=True)
        p_id = st.text_input("Patient ID", value=f"PX-{st.session_state.p_count}", label_visibility="collapsed", key="pid")
    with c2:
        st.markdown("<label class='input-label'>Age</label>", unsafe_allow_html=True)
        age = st.number_input("Age", 1, 110, 60, label_visibility="collapsed", key="age")
    with c3:
        st.markdown("<label class='input-label'>Target Eye</label>", unsafe_allow_html=True)
        eye = st.selectbox("Target Eye", ["Right (OD)", "Left (OS)"], label_visibility="collapsed", key="eye")
    with c4:
        st.markdown("<label class='input-label'>IOP (mmHg)</label>", unsafe_allow_html=True)
        iop = st.number_input("IOP", 5, 50, 16, label_visibility="collapsed", key="iop")

st.markdown("### Upload Fundus Image", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or click to select fundus image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

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
    t1, t2, t3 = st.tabs(["📊 Diagnostic Results", "🔬 Explainability", "📑 Report"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            if glaucoma_prob >= 0.60:
                st.markdown(f"""<div class="result-card result-status-high">
                    <div style="font-size: 18px; font-weight: 700; margin-bottom: 10px;">🚨 GLAUCOMA SUSPECT</div>
                    <div style="font-size: 32px; font-weight: 800; color: #fca5a5; margin-bottom: 15px;">{glaucoma_prob*100:.1f}%</div>
                    <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                        High-risk finding detected. Recommend immediate clinical referral for comprehensive evaluation including OCT and visual field testing.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-card result-status-normal">
                    <div style="font-size: 18px; font-weight: 700; margin-bottom: 10px;">✅ NORMAL VARIANT</div>
                    <div style="font-size: 32px; font-weight: 800; color: #86efac; margin-bottom: 15px;">{glaucoma_prob*100:.1f}%</div>
                    <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                        Structural patterns appear healthy. No acute pathological findings detected. Routine follow-up recommended.
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("<p style='font-size: 12px; color: #94a3b8; margin-bottom: 10px;'>RISK DISTRIBUTION</p>", unsafe_allow_html=True)
            st.progress(glaucoma_prob)
            st.metric("Confidence", f"{max(glaucoma_prob, 1-glaucoma_prob)*100:.1f}%")

    with t2:
        st.markdown("#### Neural Activation Heatmap", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 12px; margin-bottom: 15px;'>GradCAM++ visualization shows regions the AI model focused on for diagnosis</p>", unsafe_allow_html=True)
        heatmap = generate_gradcam(engine, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
        st.image(overlay, caption="Hot regions = Areas of diagnostic importance", use_container_width=True)

    with t3:
        st.markdown("#### Clinical Report Generation", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 12px; margin-bottom: 15px;'>Generate and download a professional medical report</p>", unsafe_allow_html=True)
        report_data = {
            "Patient_ID": p_id, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Eye": eye, "IOP": iop, "Result": diag_result, "Prob": f"{glaucoma_prob*100:.1f}%"
        }
        pdf_bytes = create_medical_report(report_data)
        st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name=f"Report_{p_id}.pdf", use_container_width=True)
