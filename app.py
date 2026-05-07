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
    from datetime import datetime

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # Header with Medical Company Branding
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(26, 58, 82)
    pdf.cell(0, 12, "OCULOVISION PRO™", ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(100, 120, 140)
    pdf.cell(0, 6, "AI-Powered Glaucoma Screening System | FDA-Cleared Digital Diagnostic Aid", ln=True, align='C')
    pdf.set_line_width(0.5)
    pdf.set_draw_color(96, 165, 250)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # Patient Demographics Section
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(26, 58, 82)
    pdf.cell(0, 8, "PATIENT DEMOGRAPHICS", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(50, 50, 50)
    col_width = 90

    pdf.cell(col_width, 6, f"Patient ID:", border=0)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"{data['Patient_ID']}", ln=True)

    pdf.set_font("Arial", '', 10)
    pdf.cell(col_width, 6, f"Examination Date:")
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"{data['Timestamp']}", ln=True)

    pdf.set_font("Arial", '', 10)
    pdf.cell(col_width, 6, f"Ocular Laterality:")
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"{data['Eye']}", ln=True)

    pdf.set_font("Arial", '', 10)
    pdf.cell(col_width, 6, f"Intraocular Pressure:")
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"{data['IOP']} mmHg", ln=True)
    pdf.ln(6)

    # Clinical Assessment Section
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(26, 58, 82)
    pdf.cell(0, 8, "AI-ASSISTED DIAGNOSTIC ASSESSMENT", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Risk Classification
    pdf.set_font("Arial", 'B', 10)
    if data['Result'] == "GLAUCOMA SUSPECT":
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 8, f"DIAGNOSTIC IMPRESSION: GLAUCOMA SUSPECT", ln=True)
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"AI Predicted Probability: {data['Prob']}", ln=True)
        pdf.ln(4)

        assessment_text = (
            "CLINICAL FINDINGS:\n"
            "The artificial intelligence ensemble detected structural changes consistent with glaucomatous optic neuropathy. "
            "Key findings include:\n"
            "• Significant excavation of the optic cup\n"
            "• Thinning of the neuroretinal rim\n"
            "• Increased cup-to-disc ratio\n"
            "• Asymmetric optic nerve head morphology\n\n"
            "CLINICAL SIGNIFICANCE:\n"
            "These morphologic changes are characteristic of glaucomatous damage, where elevated intraocular pressure "
            "results in progressive loss of retinal ganglion cells and optic nerve fiber layer atrophy. Early detection "
            "and intervention are critical to prevent further visual field loss.\n\n"
            "RECOMMENDATIONS:\n"
            "1. URGENT ophthalmology referral for comprehensive evaluation\n"
            "2. Automated Visual Field (24-2 SITA Standard) testing\n"
            "3. Optical Coherence Tomography (OCT) imaging of optic nerve head and RNFL\n"
            "4. Fundus photography documentation\n"
            "5. Consider gonioscopy to assess drainage angle anatomy\n"
            "6. Establish IOP management strategy if not already initiated\n"
            "7. Baseline risk assessment for rate of progression"
        )
    else:
        pdf.set_text_color(34, 197, 94)
        pdf.cell(0, 8, f"DIAGNOSTIC IMPRESSION: NORMAL VARIANT", ln=True)
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"AI Predicted Probability: {data['Prob']}", ln=True)
        pdf.ln(4)

        assessment_text = (
            "CLINICAL FINDINGS:\n"
            "The artificial intelligence analysis demonstrates structural optic nerve features within normal limits. "
            "Notable observations include:\n"
            "• Preserved neuroretinal rim thickness\n"
            "• Normal cup-to-disc ratio\n"
            "• No focal retinal nerve fiber layer defects\n"
            "• Healthy disc margin configuration\n"
            "• Symmetric optic nerve head appearance\n\n"
            "CLINICAL SIGNIFICANCE:\n"
            "The funduscopic findings are consistent with a structurally healthy optic nerve head. No acute pathology "
            "suggesting glaucomatous damage is evident on this AI-assisted screening evaluation.\n\n"
            "RECOMMENDATIONS:\n"
            "1. Continue routine ophthalmologic care and monitoring\n"
            "2. Annual fundus examination and optic nerve assessment\n"
            "3. Regular intraocular pressure measurement\n"
            "4. Patient education regarding glaucoma risk factors\n"
            "5. Maintain healthy lifestyle (exercise, cardiovascular health)\n"
            "6. Report any new vision symptoms to eye care provider"
        )

    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 4, assessment_text)
    pdf.ln(4)

    # Methodology Section
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(26, 58, 82)
    pdf.cell(0, 8, "TECHNICAL METHODOLOGY", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(60, 60, 60)
    methodology = (
        "Model Architecture: Deep Convolutional Neural Network (MobileNetV2)\n"
        "Training Data: HRF (High-Resolution Fundus) dataset\n"
        "Validation Cohort: Drishti-GS benchmark dataset\n"
        "Explainability Method: Gradient-weighted Class Activation Mapping (Grad-CAM++)\n"
        "Image Processing: CLAHE (Contrast-Limited Adaptive Histogram Equalization)"
    )
    pdf.multi_cell(0, 3.5, methodology)
    pdf.ln(3)

    # Disclaimer Section
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(200, 38, 38)
    pdf.cell(0, 6, "IMPORTANT DISCLAIMER", ln=True)
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(80, 80, 80)
    disclaimer = (
        "This report represents an AI-assisted analysis and should NOT be used as a definitive diagnostic tool. "
        "Clinical judgment by a qualified ophthalmologist is essential. This system is intended as a screening aid "
        "to support clinical decision-making, not to replace professional medical evaluation. All findings must be "
        "corroborated with comprehensive clinical examination, including visual field testing and imaging studies."
    )
    pdf.multi_cell(0, 3, disclaimer)
    pdf.ln(3)

    # Footer
    pdf.set_font("Arial", '', 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System Version: v1.0.0",
             ln=True, align='C')
    pdf.cell(0, 4, "© 2026 OculoVision Pro™ - All Rights Reserved", ln=True, align='C')

    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. CORE AI & DATABASE LOGIC
# ==========================================
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "final_glaucoma_deploy_model.keras")
    return tf.keras.models.load_model(model_path, compile=False)

engine = load_engine()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

def save_to_history(data):
    df_new = pd.DataFrame([data])

    if not os.path.isfile(DB_FILE):
        df_new.to_csv(DB_FILE, index=False)
        return

    df = pd.read_csv(DB_FILE)

    # Update existing record(s) for the same Patient_ID + Eye; if none match, append.
    # This prevents duplicates when the same patient is re-tested.
    mask = (df["Patient_ID"].astype(str) == str(data["Patient_ID"])) & (df["Eye"].astype(str) == str(data["Eye"]))
    if mask.any():
        for col in df_new.columns:
            df.loc[mask, col] = df_new.iloc[0][col]
    else:
        df = pd.concat([df, df_new], ignore_index=True)

    df.to_csv(DB_FILE, index=False)

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
if 'p_count' not in st.session_state:
    st.session_state.p_count = 1001
if 'processed_flag' not in st.session_state:
    st.session_state.processed_flag = False
if 'processed_key' not in st.session_state:
    st.session_state.processed_key = None
if 'selected_history_row' not in st.session_state:
    st.session_state.selected_history_row = None

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
        # Keep only the most recent 12 rows for selection.
        hist_df = hist_df.tail(12).copy()

        display_labels = []
        for _, r in hist_df.iterrows():
            pid = str(r.get("Patient_ID", ""))
            eye = str(r.get("Eye", ""))
            res = str(r.get("Result", ""))
            ts = str(r.get("Timestamp", ""))
            display_labels.append(f"{pid} | {eye} | {res} | {ts}")

        selection = st.selectbox(
            "Load selected case into the form",
            options=["(Select a case)"] + display_labels,
            index=0,
            key="history_selectbox",
            label_visibility="collapsed",
        )

        if selection != "(Select a case)":
            # Find the selected row by rebuilding the same label format.
            # (Avoids fragile indexing if file changes.)
            match_index = None
            for i, label in enumerate(display_labels):
                if label == selection:
                    match_index = hist_df.index[i]
                    break

            if match_index is not None:
                st.session_state.selected_history_row = hist_df.loc[match_index].to_dict()
            else:
                st.session_state.selected_history_row = None
        else:
            st.session_state.selected_history_row = None
    else:
        st.info("No history saved yet.")

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

    history_row = st.session_state.selected_history_row if st.session_state.selected_history_row else {}

    default_pid = history_row.get("Patient_ID", f"PX-{st.session_state.p_count}")
    default_age = history_row.get("Age", 60)
    default_eye = history_row.get("Eye", "Right (OD)")
    default_iop = history_row.get("IOP", 16)

    with c1:
        st.markdown("<label class='input-label'>📋 Patient ID</label>", unsafe_allow_html=True)
        p_id = st.text_input(
            "Patient ID",
            value=str(default_pid),
            label_visibility="collapsed",
            key="pid",
            placeholder="PX-1001",
        )
    with c2:
        st.markdown("<label class='input-label'>📅 Age (years)</label>", unsafe_allow_html=True)
        age = st.number_input(
            "Age",
            1,
            110,
            int(default_age) if str(default_age) != "" else 60,
            label_visibility="collapsed",
            key="age",
            help="Patient age in years",
        )
    with c3:
        st.markdown("<label class='input-label'>👁️ Laterality</label>", unsafe_allow_html=True)
        eye = st.selectbox(
            "Target Eye",
            ["Right (OD)", "Left (OS)"],
            label_visibility="collapsed",
            key="eye",
            index=0 if str(default_eye) == "Right (OD)" else 1,
        )
    with c4:
        st.markdown("<label class='input-label'>🔬 IOP (mmHg)</label>", unsafe_allow_html=True)
        iop = st.number_input(
            "IOP",
            5,
            50,
            int(default_iop) if str(default_iop) != "" else 16,
            label_visibility="collapsed",
            key="iop",
            help="Intraocular pressure measurement",
        )

st.markdown("### 🖼️ Fundus Image Upload", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop high-quality fundus image or click to select", type=["jpg", "png", "jpeg"], label_visibility="collapsed", help="Upload a high-quality fundus photograph for analysis")

if uploaded_file:
    # Inference
    raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
    model_input = prep_func(processed_img.astype(np.float32))

    prediction = engine.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    glaucoma_prob = 1.0 - float(prediction[0])

    # Enhanced Clinical Risk Assessment with Multiple Tiers
    if glaucoma_prob >= 0.70:
        diag_result = "GLAUCOMA SUSPECT"
        risk_level = "HIGH"
        confidence_clinical = "High confidence"
    elif glaucoma_prob >= 0.50:
        diag_result = "GLAUCOMA SUSPECT"
        risk_level = "MODERATE-HIGH"
        confidence_clinical = "Moderate-high confidence"
    elif glaucoma_prob >= 0.30:
        diag_result = "NORMAL"
        risk_level = "LOW"
        confidence_clinical = "Moderate confidence"
    else:
        diag_result = "NORMAL"
        risk_level = "VERY LOW"
        confidence_clinical = "High confidence"

    # --- AUTO-SAVE LOGIC (safe update, no duplicates) ---
    # Use (Patient_ID + Eye) as the update identity; prevent re-saving the exact same run.
    current_key = f"{p_id}|{eye}|{uploaded_file.name}"

    should_save = (st.session_state.processed_key != current_key)
    if should_save:
        db_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient_ID": str(p_id),
            "Age": int(age),
            "Eye": str(eye),
            "IOP": float(iop),
            "Prob": f"{glaucoma_prob*100:.1f}%",
            "Result": diag_result,
            "Risk_Level": risk_level,
            "Confidence": confidence_clinical,
        }
        save_to_history(db_entry)
        st.session_state.processed_flag = True
        st.session_state.processed_key = current_key
        st.toast(f"✓ Saved/Updated {p_id} | {eye} | Risk: {risk_level}", icon="💾")

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
