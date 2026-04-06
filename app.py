# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st  
# Streamlit → used to build interactive web dashboard UI (no frontend required)

import numpy as np  
# NumPy → used for matrix operations (images are stored as pixel arrays)

import matplotlib.pyplot as plt  
# Matplotlib → used for plotting graphs (histogram, intensity profile)

import cv2  
# OpenCV → main computer vision library
# used for:
# - grayscale conversion
# - edge detection
# - filtering
# - transformations
# - histogram equalization

from PIL import Image  
# PIL → used to load images easily and convert formats


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(page_title="Image Analysis Lab", layout="wide")
# Sets dashboard title and wide layout (better visualization)

st.title("IDSA PROJECT - Image Analysis Dashboard")
# Main heading


st.markdown("""
## 📊 Image Processing & Analysis Dashboard

This dashboard demonstrates:

### 🔹 Transformations
- Grayscale, Rotate, Flip, Transpose

### 🔹 Enhancement
- Histogram Equalization
- Brightness & Contrast

### 🔹 Filtering
- Mean, Median, Gaussian

### 🔹 Feature Detection
- Canny Edge Detection
- Sobel Edge Detection

### 🔹 Analysis
- Histogram
- FFT Spectrum
- Intensity Profile
- ROI Analysis
""")


# =========================
# SIDEBAR (INPUT CONTROLS)
# =========================

st.sidebar.header("Upload Image")

image_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png","jpg","jpeg","bmp"]
)
# Allows user to upload image → returns file object


st.sidebar.header("Analysis Tools")

show_histogram = st.sidebar.checkbox("Show Image Histogram", True)
show_fft_image = st.sidebar.checkbox("Show Image Frequency Spectrum")
show_intensity_profile = st.sidebar.checkbox("Show Intensity Profile")
show_roi_analysis = st.sidebar.checkbox("Enable ROI Analysis")


# =========================
# MAIN DASHBOARD
# =========================

st.header("Image Processing Dashboard")

if image_file:

    # =========================
    # LOAD IMAGE
    # =========================

    img = Image.open(image_file).convert("RGB")
    # Load image → convert to RGB (ensures 3 channels)

    img_np = np.array(img)
    # Convert image into NumPy array
    # Shape = (height, width, channels)

    st.image(img, caption="Original Image", use_container_width=True)

    # =========================
    # IMAGE INFO
    # =========================

    st.subheader("Image Information")
    st.write("Resolution:", img_np.shape[1], "x", img_np.shape[0])
    st.write("Channels:", img_np.shape[2])

    st.divider()

    # =========================
    # OPERATION SELECTOR
    # =========================

    operation = st.radio(
        "Select Operation",
        [
            "Grayscale","Rotate","Flip","Transpose","Matrix Info",
            "Histogram Equalization","Edge Detection","Sobel Edge",
            "Add Noise","Denoise Image","Brightness / Contrast",
            "Blur Comparison","Thresholding","Color Channels"
        ],
        horizontal=True
    )


# =========================
# GRAYSCALE
# =========================

    if operation == "Grayscale":

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Converts RGB image → grayscale (1 channel)

        col1,col2 = st.columns(2)
        col1.image(img,caption="Original")
        col2.image(gray,caption="Grayscale")


# =========================
# ROTATE
# =========================

    elif operation == "Rotate":

        angle = st.select_slider("Rotation Angle", options=[45,90,180,270])

        h,w = img_np.shape[:2]

        matrix = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        # Create rotation matrix using center + angle

        rotated = cv2.warpAffine(img_np,matrix,(w,h))
        # Apply transformation

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(rotated,caption=f"Rotated {angle}°")


# =========================
# FLIP
# =========================

    elif operation == "Flip":

        flip_type = st.radio("Flip Direction", ["Horizontal","Vertical"], horizontal=True)

        if flip_type == "Horizontal":
            flipped = np.fliplr(img_np)   # left-right flip
        else:
            flipped = np.flipud(img_np)   # up-down flip

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(flipped,caption=f"{flip_type} Flip")


# =========================
# TRANSPOSE
# =========================

    elif operation == "Transpose":

        transposed = np.transpose(img_np,(1,0,2))
        # Swap height and width

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(transposed,caption="Transposed")


# =========================
# MATRIX INFO
# =========================

    elif operation == "Matrix Info":

        st.write("Shape:", img_np.shape)
        st.write("Height:", img_np.shape[0])
        st.write("Width:", img_np.shape[1])
        st.write("Channels:", img_np.shape[2])


# =========================
# HISTOGRAM EQUALIZATION
# =========================

    elif operation == "Histogram Equalization":

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        # Improves contrast by spreading intensity values

        col1,col2 = st.columns(2)
        col1.image(gray)
        col2.image(equalized)


# =========================
# EDGE DETECTION (CANNY)
# =========================

    elif operation == "Edge Detection":

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray,100,200)
        # Multi-step edge detection algorithm

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(edges)


# =========================
# SOBEL EDGE
# =========================

    elif operation == "Sobel Edge":

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        col1,col2,col3 = st.columns(3)
        col1.image(gray)
        col2.image(sobelx)
        col3.image(sobely)


# =========================
# ADD NOISE
# =========================

    elif operation == "Add Noise":

        noise_level = st.slider("Noise Level",0,50,20)

        noise = np.random.normal(0,noise_level,img_np.shape)
        noisy = np.clip(img_np + noise,0,255).astype(np.uint8)

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(noisy)


# =========================
# DENOISE
# =========================

    elif operation == "Denoise Image":

        noise = np.random.normal(0,20,img_np.shape)
        noisy = np.clip(img_np + noise,0,255).astype(np.uint8)

        denoise = cv2.fastNlMeansDenoisingColored(noisy,None,10,10,7,21)

        col1,col2,col3 = st.columns(3)
        col1.image(img)
        col2.image(noisy)
        col3.image(denoise)


# =========================
# BRIGHTNESS & CONTRAST
# =========================

    elif operation == "Brightness / Contrast":

        alpha = st.slider("Contrast", 0.5, 3.0, 1.0)
        beta = st.slider("Brightness", -100, 100, 0)

        adjusted = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(adjusted)


# =========================
# BLUR COMPARISON
# =========================

    elif operation == "Blur Comparison":

        mean = cv2.blur(img_np,(5,5))
        gaussian = cv2.GaussianBlur(img_np,(5,5),0)
        median = cv2.medianBlur(img_np,5)

        col1,col2,col3,col4 = st.columns(4)
        col1.image(img)
        col2.image(mean)
        col3.image(gaussian)
        col4.image(median)


# =========================
# THRESHOLDING
# =========================

    elif operation == "Thresholding":

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        thresh_val = st.slider("Threshold",0,255,127)
        _, thresh = cv2.threshold(gray, thresh_val,255,cv2.THRESH_BINARY)

        col1,col2 = st.columns(2)
        col1.image(gray)
        col2.image(thresh)


# =========================
# COLOR CHANNELS
# =========================

    elif operation == "Color Channels":

        r,g,b = cv2.split(img_np)

        col1,col2,col3 = st.columns(3)
        col1.image(r)
        col2.image(g)
        col3.image(b)


# =========================
# GLOBAL ANALYSIS
# =========================

    if show_histogram:

        fig, ax = plt.subplots()

        for i,color in enumerate(["r","g","b"]):
            hist = cv2.calcHist([img_np],[i],None,[256],[0,256])
            ax.plot(hist,color=color)

        st.pyplot(fig)


    if show_fft_image:

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        spectrum = 20*np.log(np.abs(fshift)+1)

        fig, ax = plt.subplots()
        ax.imshow(spectrum, cmap="inferno")
        ax.axis("off")

        st.pyplot(fig)


    if show_intensity_profile:

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        mid_row = gray[gray.shape[0]//2]

        fig, ax = plt.subplots()
        ax.plot(mid_row)

        st.pyplot(fig)


    if show_roi_analysis:

        h,w = img_np.shape[:2]

        x = st.slider("X",0,w-1,0)
        y = st.slider("Y",0,h-1,0)

        roi_w = st.slider("Width",50,min(400,w))
        roi_h = st.slider("Height",50,min(400,h))

        roi = img_np[y:y+roi_h, x:x+roi_w]

        st.image(roi)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        fig, ax = plt.subplots()
        ax.hist(gray_roi.ravel(),bins=50)

        st.pyplot(fig)

else:
    st.info("Upload an image to begin analysis")