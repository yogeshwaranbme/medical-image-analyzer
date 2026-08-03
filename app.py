import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import filters, color, exposure
from openai import OpenAI
from gtts import gTTS
import os
import io
import base64

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
        enhanced = exposure.equalize_adapthist(self.gray_image, clip_limit=0.03)
        return enhanced

    def detect_edges(self):
        edges = filters.sobel(self.gray_image)
        return edges

    def get_statistics(self):
        return {
            "Minimum Pixel Value": float(np.min(self.gray_image)),
            "Maximum Pixel Value": float(np.max(self.gray_image)),
            "Average (Mean) Brightness": float(np.mean(self.gray_image))
        }


# =====================================================================
# 2. OPENROUTER LAB REPORT ANALYZER & MULTILINGUAL TTS CLASS (OOPs)
# =====================================================================
class LabReportAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai",
            api_key=api_key,
        )

    def _encode_image_to_base64(self, pillow_image):
        buffered = io.BytesIO()
        pillow_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_report_image(self, report_image, language_name):
        base64_image = self._encode_image_to_base64(report_image)
        
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
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501", 
                    "X-Title": "Healthcare IT Portal",       
                },
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices.message.content
        except Exception as e:
            return f"OpenRouter Cloud Processing Error: {str(e)}"

    def text_to_speech(self, text, lang_code):
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
st.set_page_config(page_title="Healthcare IT Portal", layout="wide")
st.title("🏥 Enterprise Cloud-Based Healthcare IT Platform")
st.write("A comprehensive biomedical portfolio workspace combining clinical imaging, OpenRouter AI diagnostics, and multi-language patient tracking.")

if "OPENROUTER_API_KEY" not in st.session_state:
    st.session_state["OPENROUTER_API_KEY"] = ""

with st.sidebar:
    st.header("🔑 Cloud Configuration")
    user_key = st.text_input("Enter OpenRouter API Key:", type="password", value=st.session_state["OPENROUTER_API_KEY"])
    if user_key:
        st.session_state["OPENROUTER_API_KEY"] = user_key
        st.success("OpenRouter Key activated securely for this session!")
    else:
        st.info("Please input an OpenRouter API key to enable free automated AI tracking features.")

tab1, tab2 = st.tabs(["📸 Medical Image Analyzer", "🧪 AI Lab Report Tracker & Voice Explainer"])

# ---------------------------------------------------------------------
# TAB 1: MEDICAL IMAGE ANALYZER
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
# TAB 2: LAB REPORT TRACKER & MULTILINGUAL VOICE EXPLAINER
# ---------------------------------------------------------------------
with tab2:
    st.header("Automated Medical Lab Tracker & Voice Assistance Portal")
    st.write("Track pathology metrics in memory and synthesize multilingual audio briefings using OpenRouter Free Models.")

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
        selected_lang_name = st.selectbox("Select Target Explanation Language:", list(languages.keys()), key="lang_selector")
        target_lang_code = languages[selected_lang_name]
        
        uploaded_report = st.file_uploader("Upload a screenshot or photo of lab report:", type=["png", "jpg", "jpeg"], key="lab_uploader")
        
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
            
            if not st.session_state["OPENROUTER_API_KEY"]:
                st.warning("⚠️ Action Required: Please enter your OpenRouter API Key in the sidebar configuration to trigger analytics.")
            else:
                if st.button("🚀 Analyze Report & Synthesize Audio", key="analyze_report_btn"):
                    with st.spinner("Processing medical text data and generating voice file..."):
                        
                        uploaded_report.seek(0)
                        report_image = Image.open(uploaded_report)
                        
                        analyzer = LabReportAnalyzer(st.session_state["OPENROUTER_API_KEY"])
                        
                        insights_text = analyzer.analyze_report_image(report_image, selected_lang_name)
                        
                        st.markdown(f"### 📝 Medical Breakdown ({selected_lang_name}):")
                        st.info(insights_text)
                        
                        voice_file_path = analyzer.text_to_speech(insights_text, target_lang_code)
                        
                        if voice_file_path and os.path.exists(voice_file_path):
                            st.markdown("### 🔊 Interactive Voice Assistant Playback:")
                            st.audio(voice_file_path, format="audio/mp3")
                            st.success(f"Execution complete. Output successfully processed in {selected_lang_name}.")
                            os.remove(voice_file_path)
                        else:
                            st.info("Upload a patient laboratory image file to initiate tracking modules.")
