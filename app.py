import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import filters, color, exposure
import google.generativeai as genai
from gtts import gTTS
import os

# =====================================================================
# 1. MEDICAL IMAGE PROCESSING CLASS (OOPs)
# =====================================================================
class MedicalImageProcessor:
    def __init__(self, image):
        # Convert PIL Image to NumPy array for digital signal/image processing
        self.original_image = np.array(image)
        
        # Convert to Grayscale if the uploaded image is RGB (standard for medical processing)
        if len(self.original_image.shape) == 3:
            self.gray_image = color.rgb2gray(self.original_image)
        else:
            self.gray_image = self.original_image

    def enhance_contrast(self):
        """Implements Contrast Limited Adaptive Histogram Equalization (CLAHE) for better visibility"""
        enhanced = exposure.equalize_adapthist(self.gray_image, clip_limit=0.03)
        return enhanced

    def detect_edges(self):
        """Applies a Sobel Filter to highlight structural boundaries like fractures or tumors"""
        edges = filters.sobel(self.gray_image)
        return edges

    def get_statistics(self):
        """Extracts pixel statistics for basic healthcare data analytics"""
        stats = {
            "Minimum Pixel Value": float(np.min(self.gray_image)),
            "Maximum Pixel Value": float(np.max(self.gray_image)),
            "Average (Mean) Brightness": float(np.mean(self.gray_image))
        }
        return stats


# =====================================================================
# 2. LAB REPORT ANALYZER & MULTILINGUAL VOICE CLASS (OOPs Add-on)
# =====================================================================
class LabReportAnalyzer:
    def __init__(self, api_key):
        """Configures the free Google Gemini Multimodal AI Engine"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_report_image(self, report_image, language_name):
        """Uses computer vision and NLP to extract text and analyze data in a target language"""
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
            return f"Cloud AI processing error: {str(e)}"

    def text_to_speech(self, text, language_code):
        """Converts analyzed medical text summaries into natural speech (.mp3)"""
        try:
            # Strip out markdown symbols before feeding into the voice reader engine
            clean_text = text.replace("*", "").replace("#", "")
            tts = gTTS(text=clean_text, lang=language_code, slow=False)
            
            temp_filename = "temp_voice_report.mp3"
            tts.save(temp_filename)
            return temp_filename
        except Exception as e:
            st.error(f"Voice Synthesis Engine failed: {str(e)}")
            return None


# =====================================================================
# 3. STREAMLIT WEB UI CONFIGURATION & NAVIGATION TABS
# =====================================================================
st.set_page_config(page_title="Healthcare IT: Portal", layout="wide")
st.title("🏥 Enterprise Cloud-Based Healthcare IT Platform")
st.write("A comprehensive biomedical portfolio workspace combining clinical imaging, AI diagnostics, and multi-language patient tracking.")

# Session state setup to handle API keys safely across app re-runs
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = ""

with st.sidebar:
    st.header("🔑 Cloud Configuration")
    user_key = st.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state["GEMINI_API_KEY"])
    if user_key:
        st.session_state["GEMINI_API_KEY"] = user_key
        st.success("API Key activated securely for this session!")
    else:
        st.info("Please input an API key to enable AI-powered automated tracking features.")

# Use Streamlit tabs to organize the software cleanly
tab1, tab2 = st.tabs(["📸 Medical Image Analyzer", "🧪 AI Lab Report Tracker & Voice Explainer"])

# ---------------------------------------------------------------------
# TABS 1: ORIGINAL MEDICAL IMAGE ANALYSIS ARCHITECTURE
# ---------------------------------------------------------------------
with tab1:
    st.header("Digital Image Scan Processor")
    uploaded_file = st.file_uploader("Upload a medical scan (X-ray, MRI, CT)...", type=["png", "jpg", "jpeg"], key="img_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
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
            operation = st.selectbox("Select an Image Processing filter:", ["Contrast Enhancement", "Edge Detection"], key="filter_select")
            
            if operation == "Contrast Enhancement":
                processed_img = processor.enhance_contrast()
                st.image(processed_img, caption="Enhanced Contrast Scan", use_container_width=True)
                st.success("Cloud Execution Successful: Histogram equalization applied.")
                
            elif operation == "Edge Detection":
                processed_img = processor.detect_edges()
                st.image(processed_img, caption="Detected Structural Edges", use_container_width=True)
                st.success("Cloud Execution Successful: Sobel filter applied.")

# ---------------------------------------------------------------------
# TAB 2: BRAND NEW ADDITION - LAB REPORT TRACKER & MULTILINGUAL VOICE EXPLAINER
# ---------------------------------------------------------------------
with tab2:
    st.header("Automated Medical Lab Tracker & Voice Assistance Portal")
    st.write("Track pathology metrics in memory and synthesize multilingual audio briefings for patient accessibility.")

    # Dictionary mapping spoken name to gTTS standard ISO language codes
    languages = {
        "English": "en",
        "Punjabi (ਪੰਜਾਬੀ)": "pa",
        "Hindi (हिन्दी)": "hi",
        "Spanish (Español)": "es",
        "French (Français)": "fr"
    }

    col_lab1, col_lab2 = st.columns(2)

    with col_lab1:
        st.subheader("📥 Data Tracking Input")
        selected_lang_name = st.selectbox("Select Target Explanation Language:", list(languages.keys()))
        target_lang_code = languages[selected_lang_name]
        
        uploaded_report = st.file_uploader("Upload a screenshot or photo of lab report:", type=["png", "jpg", "jpeg"], key="lab_uploader")
        
        # Visual metrics mimicking live database schema storage triggers
        st.subheader("🗄️ Backend Cloud-Database Tracker")
        if uploaded_report:
            st.metric(label="File Storage Status", value="Buffered in Cloud RAM")
            st.metric(label="Simulated SQL Sync State", value="Locked & Awaiting Analytics")
        else:
            st.metric(label="File Storage Status", value="Empty/Idle")

    with col_lab2:
        st.subheader("🤖 AI Diagnostic & Accessibility Outputs")
        
        if uploaded_report is not None:
            st.image(uploaded_report, caption="Buffered Document Payload", width=280)
            
            # Block processing if API key is missing
            if not st.session_state["GEMINI_API_KEY"]:
                st.warning("⚠️ Action Required: Please enter your Gemini API Key in the sidebar configuration to trigger analytics.")
            else:
                if st.button("🚀 Analyze Report & Synthesize Audio"):
                    with st.spinner("Processing medical text data and generating voice file..."):
                        
                        # Load file and initialize analyzer object
                        report_image = Image.open(uploaded_report)
                        analyzer = LabReportAnalyzer(st.session_state["GEMINI_API_KEY"])
                        
                        # Step 1: Text analysis in the chosen language 
                        insights_text = analyzer.analyze_report_image(report_image, selected_lang_name)
                        
                        # Step 2: Show translated medical breakdown on screen
                        st.markdown(f"### 📝 Medical Breakdown ({selected_lang_name}):")
                        st.info(insights_text)
                        
                        # Step 3: Convert text output to high-fidelity audio
                        voice_file_path = analyzer.text_to_speech(insights_text, target_lang_code)
                        
                        if voice_file_path and os.path.exists(voice_file_path):
                            st.markdown("### 🔊 Interactive Voice Assistant Playback:")
