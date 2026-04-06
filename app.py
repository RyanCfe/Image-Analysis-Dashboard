# =========================
# IMPORTS
# =========================

import streamlit as st          # UI framework
import numpy as np             # for matrix/image operations
import matplotlib.pyplot as plt # for graphs
import cv2                     # OpenCV for image processing
from PIL import Image          # image loading


# =========================
# PAGE SETUP
# =========================

st.set_page_config(page_title="Image Analysis Lab", layout="wide")

st.title("Image Analysis Dashboard")

st.markdown("""
Upload an image and explore different image processing techniques interactively.
""")


# =========================
# SIDEBAR INPUT
# =========================

st.sidebar.header("Upload Image")

image_file = st.sidebar.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp"]
)

st.sidebar.header("Analysis Options")

show_histogram = st.sidebar.checkbox("Histogram", True)
show_fft_image = st.sidebar.checkbox("FFT Spectrum")
show_intensity_profile = st.sidebar.checkbox("Intensity Profile")
show_roi_analysis = st.sidebar.checkbox("ROI Analysis")


# =========================
# MAIN DASHBOARD
# =========================

st.header("Processing")

if image_file:

    # -------------------------
    # LOAD IMAGE
    # -------------------------

    img = Image.open(image_file).convert("RGB")  
    # convert to RGB → ensures consistent 3 channels

    img_np = np.array(img)  
    # convert image → numpy array (height, width, channels)

    st.image(img, caption="Original Image", use_container_width=True)

    # -------------------------
    # IMAGE INFO
    # -------------------------

    st.subheader("Image Info")

    st.write(f"Resolution: {img_np.shape[1]} x {img_np.shape[0]}")
    st.write(f"Channels: {img_np.shape[2]}")

    st.divider()

    # -------------------------
    # OPERATION SELECTOR
    # -------------------------

    operation = st.radio(
        "Choose Operation",
        [
            "Grayscale","Rotate","Flip","Transpose","Matrix Info",
            "Histogram Equalization","Edge Detection","Sobel Edge",
            "Add Noise","Denoise Image","Brightness / Contrast",
            "Blur Comparison","Thresholding","Color Channels"
        ],
        horizontal=True
    )

    # =========================
    # OPERATIONS
    # =========================

    if operation == "Grayscale":

        st.info("Grayscale converts the image into a single intensity channel.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(gray)


    elif operation == "Rotate":

        st.info("Image is rotated using a transformation matrix around its center.")

        angle = st.select_slider("Angle", [45, 90, 180, 270])

        h, w = img_np.shape[:2]

        # rotation matrix centered at image center
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

        # apply rotation
        rotated = cv2.warpAffine(img_np, matrix, (w, h))

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(rotated)


    elif operation == "Flip":

        st.info("Flipping mirrors the image horizontally or vertically.")

        direction = st.radio("Direction", ["Horizontal", "Vertical"], horizontal=True)

        flipped = np.fliplr(img_np) if direction == "Horizontal" else np.flipud(img_np)

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(flipped)


    elif operation == "Transpose":

        st.info("Transpose swaps image axes (width becomes height).")

        # swap axes
        transposed = np.transpose(img_np, (1, 0, 2))

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(transposed)


    elif operation == "Matrix Info":

        st.info("Displays raw matrix structure of the image.")

        st.write("Shape:", img_np.shape)
        st.write("Height:", img_np.shape[0])
        st.write("Width:", img_np.shape[1])


    elif operation == "Histogram Equalization":

        st.info("Improves contrast by spreading pixel intensity values.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # spreads intensity values → better contrast
        eq = cv2.equalizeHist(gray)

        col1, col2 = st.columns(2)
        col1.image(gray)
        col2.image(eq)


    elif operation == "Edge Detection":

        st.info("Canny detects edges using gradients and thresholding.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # thresholds define edge sensitivity
        edges = cv2.Canny(gray, 100, 200)

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(edges)


    elif operation == "Sobel Edge":

        st.info("Sobel detects edges in horizontal and vertical directions.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # gradient in x direction
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

        # gradient in y direction
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        col1, col2, col3 = st.columns(3)
        col1.image(gray)
        col2.image(sobelx)
        col3.image(sobely)


    elif operation == "Add Noise":

        st.info("Adds Gaussian noise to simulate distortion.")

        level = st.slider("Noise Level", 0, 50, 20)

        # generate noise
        noise = np.random.normal(0, level, img_np.shape)

        # clip to valid range (0–255)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(noisy)


    elif operation == "Denoise Image":

        st.info("Removes noise while preserving important details.")

        noise = np.random.normal(0, 20, img_np.shape)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        # advanced denoising algorithm
        clean = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)

        col1, col2, col3 = st.columns(3)
        col1.image(img)
        col2.image(noisy)
        col3.image(clean)


    elif operation == "Brightness / Contrast":

        st.info("Adjusts brightness and contrast using linear scaling.")

        alpha = st.slider("Contrast", 0.5, 3.0, 1.0)
        beta = st.slider("Brightness", -100, 100, 0)

        # new_pixel = alpha * pixel + beta
        adjusted = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)

        col1, col2 = st.columns(2)
        col1.image(img)
        col2.image(adjusted)


    elif operation == "Blur Comparison":

        st.info("Different filters smooth the image in different ways.")

        mean = cv2.blur(img_np, (5, 5))
        gauss = cv2.GaussianBlur(img_np, (5, 5), 0)
        median = cv2.medianBlur(img_np, 5)

        col1, col2, col3, col4 = st.columns(4)
        col1.image(img)
        col2.image(mean)
        col3.image(gauss)
        col4.image(median)


    elif operation == "Thresholding":

        st.info("Converts image into binary using a threshold value.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        t = st.slider("Threshold", 0, 255, 127)

        _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        col1, col2 = st.columns(2)
        col1.image(gray)
        col2.image(thresh)


    elif operation == "Color Channels":

        st.info("Splits image into Red, Green, and Blue components.")

        r, g, b = cv2.split(img_np)

        col1, col2, col3 = st.columns(3)
        col1.image(r)
        col2.image(g)
        col3.image(b)


    # =========================
    # ANALYSIS
    # =========================

    if show_histogram:

        st.info("Histogram shows distribution of pixel intensities.")

        fig, ax = plt.subplots()

        for i, c in enumerate(["r", "g", "b"]):
            hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
            ax.plot(hist, color=c)

        st.pyplot(fig)


    if show_fft_image:

        st.info("FFT converts image into frequency domain.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        spectrum = 20 * np.log(np.abs(fshift) + 1)

        fig, ax = plt.subplots()
        ax.imshow(spectrum, cmap="inferno")
        ax.axis("off")

        st.pyplot(fig)


    if show_intensity_profile:

        st.info("Shows pixel intensity variation across the image.")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        row = gray[gray.shape[0] // 2]

        fig, ax = plt.subplots()
        ax.plot(row)

        st.pyplot(fig)


    if show_roi_analysis:

        st.info("ROI allows analysis of a selected part of the image.")

        h, w = img_np.shape[:2]

        x = st.slider("X", 0, w-1, 0)
        y = st.slider("Y", 0, h-1, 0)

        rw = st.slider("Width", 50, min(400, w))
        rh = st.slider("Height", 50, min(400, h))

        # extract region
        roi = img_np[y:y+rh, x:x+rw]

        st.image(roi)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        fig, ax = plt.subplots()
        ax.hist(gray_roi.ravel(), bins=50)

        st.pyplot(fig)

else:
    st.info("Upload an image to start")