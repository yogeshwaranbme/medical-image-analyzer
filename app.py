import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import filters, color, exposure
import google.generativeai as genai
from gtts import gTTS
import os
import io

# =====================================================================
# 1. MEDICAL IMAGE PROCESSING CLASS (OOPs)
# =====================================================================
class MedicalImageProcessor:
    def __init__(self, image):
        self.original_image = np.array(image)
        if len(self.original_image.shape) == 3:
            self.gray_image = color.rgb2gray(self.original_image)
        else:
            self.gray_image = self.original_image

    def enhance_contrast(self):
        """Implements Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
        enhanced = exposure.equalize_adapthist(self.gray_image, clip_limit=0.03)
        return enhanced

    def detect_edges(self):
        """Applies a Sobel Filter to highlight structural boundaries"""
        edges = filters.sobel(self.gray_image)
        return edges

    def get_statistics(self):
        """Extracts pixel statistics for basic healthcare data analytics"""
        return {
            "Minimum Pixel Value": float(np.min(self.gray_image)),
            "Maximum Pixel Value": float(np.max(self.gray_image)),
            "Average (Mean) Brightness": float(np.mean(self.gray_image))
        }


# =====================================================================
# 2. LAB REPORT ANALYZER & MULTILINGUAL TTS CLASS (OOPs)
# =====================================================================
class LabReportAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_report_image(self, report_image, language_name):
        """Uses Gemini 2.5 Flash Model to analyze medical reports"""
        prompt = f"""
        You are an expert Healthcare IT Medical AI Assistant. 
        Analyze this uploaded laboratory medical report image carefully:
        1. Identify key test parameters (e.g., Blood Sugar, Cholesterol, Hemoglobin).
        2. Point out any out-of-range (high or low) values clearly.
        3. Explain what these results mean conceptually using easy layman terms.
        
        CRITICAL CORE REQUIREMENT: Write your entire analytical response strictly and completely in {language_name} language. 
        Do not use complicated medical jargon. Keep paragraphs short and concise.
        """
        try:
            response = self.model.generate_content([prompt, report_image])
            return response.text
        except Exception as e:
            return f"Gemini Cloud Processing Error: {str(e)}"

    def text_to_speech(self, text, lang_code):
        """Converts analyzed medical text into a voice file using gTTS"""
        try:
            clean_text = text.replace("*", "").replace("#", "")
            tts = gTTS(text=clean_text, lang=lang_code, slow=False)
            audio_file = "temp_report_voice.mp3"
            tts.save(audio_file)
            return audio_file
        except Exception as e:
            st.error(f"Voice generation failed: {str(e)}")
            return None


# =====================================================================
# 3. STREAMLIT WEB UI CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Healthcare IT: Advanced Patient Portal", layout="wide")
st.title(" 🏥 Enterprise Cloud-Based Healthcare IT Platform")
st.write("A comprehensive Biomedical Engineering platform featuring Medical Imaging, AI Lab Reports, and Accessibility Tools.")

if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = ""

with st.sidebar:
    st.header("🔑 Cloud Configuration")
    user_key = st.text_input("Enter Gemini API Key:", type="password", value=st.session_state["GEMINI_API_KEY"])
    if user_key:
        st.session_state["GEMINI_API_KEY"] = user_key
        st.success("API Key saved securely for this session!")
    else:
        st.info("Please provide a Gemini API Key to unlock AI features.")

tab1, tab2 = st.tabs(["📸 Medical Image Analyzer", "🧪 AI Lab Report Tracker & Voice Explainer"])

# ---------------------------------------------------------------------
# TAB 1: MEDICAL IMAGE ANALYZER
# ---------------------------------------------------------------------
with tab1:
    st.header("Medical Image Scan Processor")
    uploaded_image = st.file_uploader("Upload a medical scan (X-ray, MRI, CT)...", type=["png", "jpg", "jpeg"], key="img_up")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        processor = MedicalImageProcessor(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original Medical Scan")
            st.image(image, use_container_width=True)
            
            st.subheader("📊 Diagnostic Image Analytics")
            stats = processor.get_statistics()
            for key, value in stats.items():
                st.metric(label=key, value=f"{value:.4f}")

        with col2:
            st.subheader("⚙️ Cloud Processing Panel")
            operation = st.selectbox("Select an Image Processing filter:", ["Contrast Enhancement", "Edge Detection"], key="filter_sel")
            
            if operation == "Contrast Enhancement":
                processed_img = processor.enhance_contrast()
                st.image(processed_img, caption="Enhanced Contrast Scan", use_container_width=True)
                st.success("Cloud Execution Successful: Histogram equalization applied.")
            elif operation == "Edge Detection":
                processed_img = processor.detect_edges()
                st.image(processed_img, caption="Detected Structural Edges", use_container_width=True)
                st.success("Cloud Execution Successful: Sobel filter applied.")

# ---------------------------------------------------------------------
# TAB 2: AI LAB REPORT TRACKER & VOICE EXPLAINER
# ---------------------------------------------------------------------
with tab2:
    st.header("AI Laboratory Report Tracking & Translation System")
    st.write("Upload a lab report image to get instant multilingual text and audio explanations.")

    languages = {
        "English": "en",
        "Punjabi (ਪੰਜਾਬੀ)": "pa",
        "Hindi (हिन्दी)": "hi",
        "Spanish (Español)": "es",
        "French (Français)": "fr"
    }

    col_lab1, col_lab2 = st.columns()

    with col_lab1:
        st.subheader("📥 Input Panel")
        selected_lang_name = st.selectbox("Select Explanation Language:", list(languages.keys()), key="lang_selector")
        lang_code = languages[selected_lang_name]
        
        uploaded_report = st.file_uploader("Upload Lab Report Scan/Screenshot:", type=["png", "jpg", "jpeg"], key="lab_up")
        
        st.subheader("🗄️ Database Tracking Status")
        if uploaded_report:
            st.metric(label="File Status", value="Uploaded & In-Memory")
            st.metric(label="SQL Transaction Sync", value="Pending AI Output")
        else:
            st.metric(label="File Status", value="No File Tracked")

    with col_lab2:
        st.subheader("🤖 AI Diagnostic & Accessibility Output")
        
        if uploaded_report is not None:
            st.image(uploaded_report, caption="Tracked Lab Report Image", width=300)
            
            if not st.session_state["GEMINI_API_KEY"]:
                st.warning("⚠️ Action Required: Please enter your Gemini API Key in the sidebar configuration to trigger analytics.")
            else:
                if st.button("🚀 Analyze Report & Synthesize Audio", key="analyze_report_btn"):
                    with st.spinner("Processing medical text data and generating voice file..."):
                        
                        uploaded_report.seek(0)
                        report_image = Image.open(uploaded_report)
                        
                        analyzer = LabReportAnalyzer(st.session_state["GEMINI_API_KEY"])
                        
                        insights_text = analyzer.analyze_report_image(report_image, selected_lang_name)
                        
                        st.markdown(f"### 📝 Medical Breakdown ({selected_lang_name}):")
                        st.info(insights_text)
                        
                        audio_path = analyzer.text_to_speech(insights_text, lang_code)
                        
                        if audio_path and os.path.exists(audio_path):
                            st.markdown("### 🔊 Interactive Voice Assistant Playback:")
                            st.audio(audio_path, format="audio/mp3")
                            st.success(f"Success! Report completely generated in {selected_lang_name}.")
                            os.remove(audio_path)
        else:
            st.info("Please upload a laboratory report image to initiate cloud-based AI analytics tracking.")
