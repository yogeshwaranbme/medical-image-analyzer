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
            # Check if response is a dictionary or an object with attributes
            if hasattr(response, 'choices'):
                return response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            else:
                return str(response)
                
        except Exception as e:
            return f"OpenRouter Cloud Processing Error: {str(e)}"
