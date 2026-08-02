import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import filters, color, exposure

# Object-Oriented Programming (OOPs) implementation for Medical Image Processing
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

# Streamlit Web UI Configuration
st.set_page_config(page_title="Healthcare IT: Medical Image Analyzer", layout="wide")
st.title(" 🏥 Cloud-Based Medical Image Viewer & Analyzer")
st.write("A core Biomedical Engineering & Healthcare IT project running entirely in the cloud.")

# File uploader acts as our cloud storage trigger (processes image in memory without local hardware dependencies)
uploaded_file = st.file_uploader("Upload a medical scan (X-ray, MRI, CT)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded file
    image = Image.open(uploaded_file)
    
    # Instantiate the OOPs class object
    processor = MedicalImageProcessor(image)
    
    # Split UI into two dynamic columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Medical Scan")
        st.image(image, use_container_width=True)
        
        # Display extracted data metrics
        st.subheader("📊 Diagnostic Image Analytics")
        stats = processor.get_statistics()
        for key, value in stats.items():
            st.metric(label=key, value=f"{value:.4f}")

    with col2:
        st.subheader("⚙️ Cloud Processing Panel")
        
        # User dropdown selection for the processing filter
        operation = st.selectbox("Select an Image Processing filter:", ["Contrast Enhancement", "Edge Detection"])
        
        if operation == "Contrast Enhancement":
            processed_img = processor.enhance_contrast()
            st.image(processed_img, caption="Enhanced Contrast Scan", use_container_width=True)
            st.success("Cloud Execution Successful: Histogram equalization applied.")
            
        elif operation == "Edge Detection":
            processed_img = processor.detect_edges()
            st.image(processed_img, caption="Detected Structural Edges", use_container_width=True)
            st.success("Cloud Execution Successful: Sobel filter applied.")

