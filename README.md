# 🏥 Cloud-Based Medical Image Viewer & Analyzer

A cloud-native web application designed for processing and analyzing medical scans (X-rays, MRIs, CT scans). This project serves as a bridge between **Biomedical Engineering** domain knowledge and **Healthcare IT** implementation, running entirely in the cloud without requiring local hardware infrastructure.

🚀 **Live Demo:** `[Insert your Streamlit app link here]`

---

## 🌟 Key Features
* **Cloud-Based DICOM/Image Processing:** Instant rendering and analysis of medical images in JPG/PNG formats.
* **Contrast Enhancement (CLAHE):** Implements Contrast Limited Adaptive Histogram Equalization to improve the visibility of dense tissues and bone structures in X-rays/MRIs.
* **Edge Detection (Sobel Filtering):** Employs mathematical signal processing filters to highlight structural boundaries, aiding in fracture and tumor identification.
* **Diagnostic Analytics:** Automatically calculates and displays real-time pixel-level statistical metrics (Mean, Min, Max values) for diagnostic data profiling.

---

## 🛠️ Technical Stack & Skills Demonstrated

### 1. Programming & Core Concepts
* **Python:** Core language used for backend processing logic.
* **Object-Oriented Programming (OOPs):** Built using an isolated, reusable `MedicalImageProcessor` class to ensure clean, scalable, and maintainable enterprise-level code.

### 2. Domain Knowledge (Biomedical Engineering)
* **Digital Image Processing (DIP):** Applied spatial-domain filtering and contrast adjustment algorithms derived from the academic syllabus.
* **Clinical Data Analytics:** Extracted pixel statistics essential for medical imaging informatics.

### 3. Tools & Cloud Hosting
* **Streamlit Community Cloud:** Utilized for serverless cloud hosting and instant UI deployment.
* **Git/GitHub:** Version control and source code management.
* **Libraries:** `NumPy` (array manipulation), `OpenCV` & `Scikit-Image` (computer vision algorithms), `Pillow` (image handling).

---

## 📦 Project Structure
```text
├── app.py              # Main Streamlit web application & OOPs logic
├── requirements.txt    # List of cloud-dependencies and Python libraries
└── README.md           # Project documentation and overview
```

---

## ⚙️ Installation & Local Setup

If you wish to run this project locally, follow these simple steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd medical-image-analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## 🎓 Author
YOGESHWARAN RAJINEESH- Final Year Biomedical Engineering Student
LinkedIn: `https://www.linkedin.com/in/yogeshwaran-rajineesh-269b903b7?utm_source=share_via&utm_content=profile&utm_medium=member_android`
