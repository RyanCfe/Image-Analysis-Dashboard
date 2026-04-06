# Image Analysis Dashboard

An interactive image processing dashboard built using **Streamlit**, **OpenCV**, and **NumPy**.

This project allows users to explore core computer vision concepts in real-time with visual outputs and explanations.

---

## Features

### Transformations
- Grayscale Conversion  
- Image Rotation (45°, 90°, 180°, 270°)  
- Horizontal & Vertical Flip  
- Transpose  

### Image Enhancement
- Histogram Equalization (contrast improvement)  
- Brightness & Contrast Adjustment  

### Filtering Techniques
- Mean Filter  
- Gaussian Blur  
- Median Filter  

### Edge Detection
- Canny Edge Detection  
- Sobel Edge Detection  

### Advanced Analysis
- RGB Histogram Visualization  
- FFT Frequency Spectrum  
- Intensity Profile  
- ROI (Region of Interest) Analysis  

---

## What This Project Demonstrates

- Image representation using matrices (NumPy)
- Spatial vs Frequency domain analysis (FFT)
- Noise addition & removal techniques
- Edge detection using gradient-based methods
- Interactive UI-based data visualization

---

## Tech Stack

- **Python**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Matplotlib**

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
