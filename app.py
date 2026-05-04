import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile


# -------------------------
# OCR Loader
# -------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


# -------------------------
# Geometry Helpers
# -------------------------
def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]      # top-left
    rect[2] = points[np.argmax(s)]      # bottom-right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]   # top-right
    rect[3] = points[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, points):
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


# -------------------------
# Image Quality Analyzer
# -------------------------
def analyze_image_quality(gray):
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = np.mean(gray)
    contrast_score = np.std(gray)

    quality_score = 0
    warnings = []

    if blur_score >= 120:
        quality_score += 35
    elif blur_score >= 60:
        quality_score += 25
        warnings.append("Image is slightly blurry.")
    else:
        quality_score += 10
        warnings.append("Image is too blurry.")

    if 80 <= brightness_score <= 190:
        quality_score += 35
    elif brightness_score < 80:
        quality_score += 15
        warnings.append("Image is too dark.")
    else:
        quality_score += 15
        warnings.append("Image is too bright / overexposed.")

    if contrast_score >= 45:
        quality_score += 30
    elif contrast_score >= 25:
        quality_score += 20
        warnings.append("Image has medium contrast.")
    else:
        quality_score += 10
        warnings.append("Image has low contrast.")

    if quality_score >= 80:
        status = "Good"
    elif quality_score >= 55:
        status = "Usable"
    else:
        status = "Poor"

    return {
        "blur_score": round(float(blur_score), 2),
        "brightness_score": round(float(brightness_score), 2),
        "contrast_score": round(float(contrast_score), 2),
        "quality_score": quality_score,
        "status": status,
        "warnings": warnings,
    }


# -------------------------
# Scanner Pipeline Helpers
# -------------------------
def find_document_contour(
    image_bgr,
    canny_lower,
    canny_upper,
    blur_kernel,
    top_n_contours,
    min_document_area_ratio,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (blur_kernel, blur_kernel),
        0
    )

    edges = cv2.Canny(
        blurred,
        canny_lower,
        canny_upper
    )

    # -------------------------
    # Strategy 1: Canny contours
    # -------------------------
    edge_contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # -------------------------
    # Strategy 2: Threshold-based paper mask
    # Useful when document is light/white on darker background
    # -------------------------
    _, paper_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    paper_contours, _ = cv2.findContours(
        paper_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Combine both contour strategies
    all_contours = list(edge_contours) + list(paper_contours)

    contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:top_n_contours]

    document_contour = None
    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    candidate_details = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / image_area

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        candidate_details.append(
            {
                "area": round(float(contour_area), 2),
                "area_ratio": round(float(area_ratio), 4),
                "corner_count": len(approx),
            }
        )

        # Ignore tiny rectangles like table cells, rows, or boxes
        if area_ratio < min_document_area_ratio:
            continue

        # Ideal case: contour itself has 4 corners
        if len(approx) == 4:
            document_contour = approx
            break

        # Backup case:
        # If contour is large but not exactly 4 corners,
        # use minimum-area rectangle around it.
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype("int32")

        box_area = cv2.contourArea(box)
        box_area_ratio = box_area / image_area

        if box_area_ratio >= min_document_area_ratio:
            document_contour = box.reshape(4, 1, 2)
            break

    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "paper_mask": paper_mask,
        "contours": contours,
        "document_contour": document_contour,
        "candidate_details": candidate_details,
    }


def create_enhanced_scan(warped_bgr, mode, adaptive_block_size, adaptive_c):
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    if mode == "Black & White Scan":
        enhanced = cv2.adaptiveThreshold(
            warped_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size,
            adaptive_c
        )
        return enhanced

    if mode == "Grayscale Scan":
        return warped_gray

    if mode == "Contrast Enhanced":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(warped_gray)
        return enhanced

    if mode == "Sharpened":
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        )
        sharpened = cv2.filter2D(warped_gray, -1, kernel)
        return sharpened

    if mode == "Denoised + Sharpened":
        denoised = cv2.fastNlMeansDenoising(warped_gray, None, 10, 7, 21)

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        )

        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened

    return warped_gray


def run_ocr(reader, image):
    results = reader.readtext(image)

    rows = []
    extracted_lines = []

    for _, text, confidence in results:
        rows.append(
            {
                "Text": text,
                "Confidence": round(float(confidence), 3),
            }
        )
        extracted_lines.append(text)

    extracted_text = "\n".join(extracted_lines)

    if rows:
        avg_confidence = round(
            sum(row["Confidence"] for row in rows) / len(rows),
            3
        )
    else:
        avg_confidence = 0.0

    return extracted_text, rows, avg_confidence


def image_to_png_bytes(image):
    success, encoded_image = cv2.imencode(".png", image)

    if not success:
        return None

    return encoded_image.tobytes()


def grayscale_to_pil_rgb(image):
    if len(image.shape) == 2:
        return Image.fromarray(image).convert("RGB")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb).convert("RGB")


def create_zip_file(scanned_outputs):
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in scanned_outputs:
            png_bytes = image_to_png_bytes(item["enhanced_scan"])

            if png_bytes:
                zip_file.writestr(
                    f"{item['base_name']}_enhanced_scan.png",
                    png_bytes
                )

            zip_file.writestr(
                f"{item['base_name']}_ocr_text.txt",
                item["extracted_text"]
            )

    zip_buffer.seek(0)
    return zip_buffer


def create_pdf_file(scanned_outputs):
    pil_images = []

    for item in scanned_outputs:
        pil_image = grayscale_to_pil_rgb(item["enhanced_scan"])
        pil_images.append(pil_image)

    if not pil_images:
        return None

    pdf_buffer = BytesIO()

    first_image = pil_images[0]
    remaining_images = pil_images[1:]

    first_image.save(
        pdf_buffer,
        format="PDF",
        save_all=True,
        append_images=remaining_images
    )

    pdf_buffer.seek(0)
    return pdf_buffer


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Smart Document Scanner Pro",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Smart Document Scanner Pro + OCR")
st.write(
    "Upload one or multiple document images to analyze quality, scan, straighten, enhance, extract text, and export results."
)


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Scanner Controls")

show_debug = st.sidebar.checkbox("Show debug steps", value=False)

run_ocr_enabled = st.sidebar.checkbox("Run OCR", value=True)

allow_fallback = st.sidebar.checkbox(
    "Use fallback if boundary detection fails",
    value=True
)

top_n_contours = st.sidebar.slider(
    "Top contours to inspect",
    min_value=3,
    max_value=30,
    value=20,
    step=1
)

min_document_area_ratio = st.sidebar.slider(
    "Minimum document area ratio",
    min_value=0.05,
    max_value=0.80,
    value=0.20,
    step=0.05,
    help="Ignore 4-corner contours smaller than this ratio of the full image. Useful for avoiding table cells/rows."
)

canny_lower = st.sidebar.slider(
    "Canny Lower Threshold",
    min_value=0,
    max_value=255,
    value=75,
    step=5
)

canny_upper = st.sidebar.slider(
    "Canny Upper Threshold",
    min_value=0,
    max_value=255,
    value=200,
    step=5
)

blur_kernel = st.sidebar.selectbox(
    "Blur Kernel Size",
    options=[3, 5, 7, 9],
    index=1
)

adaptive_block_size = st.sidebar.selectbox(
    "Adaptive Threshold Block Size",
    options=[11, 15, 21, 31, 41],
    index=0
)

adaptive_c = st.sidebar.slider(
    "Adaptive Threshold Constant",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)

enhancement_mode = st.sidebar.selectbox(
    "Enhancement Mode",
    options=[
        "Black & White Scan",
        "Grayscale Scan",
        "Contrast Enhanced",
        "Sharpened",
        "Denoised + Sharpened",
    ],
)


uploaded_files = st.file_uploader(
    "Upload document image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# -------------------------
# Main Processing
# -------------------------
if uploaded_files:
    scanned_outputs = []

    reader = None

    if run_ocr_enabled:
        with st.spinner("Loading OCR model..."):
            reader = load_ocr_reader()

    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        base_name = uploaded_file.name.rsplit(".", 1)[0]

        st.divider()
        st.header(f"Document {file_index}: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        scanner_result = find_document_contour(
            image_bgr=image_bgr,
            canny_lower=canny_lower,
            canny_upper=canny_upper,
            blur_kernel=blur_kernel,
            top_n_contours=top_n_contours,
            min_document_area_ratio=min_document_area_ratio,
        )

        gray = scanner_result["gray"]
        blurred = scanner_result["blurred"]
        edges = scanner_result["edges"]
        paper_mask = scanner_result["paper_mask"]
        contours = scanner_result["contours"]
        document_contour = scanner_result["document_contour"]
        candidate_details = scanner_result["candidate_details"]

        quality_report = analyze_image_quality(gray)

        st.subheader("Image Quality Report")

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)

        q_col1.metric("Overall Quality", f"{quality_report['quality_score']}/100")
        q_col2.metric("Status", quality_report["status"])
        q_col3.metric("Blur Score", quality_report["blur_score"])
        q_col4.metric("Contrast", quality_report["contrast_score"])

        with st.expander("Detailed quality metrics"):
            st.write("Brightness Score:", quality_report["brightness_score"])
            st.write("Blur Score:", quality_report["blur_score"])
            st.write("Contrast Score:", quality_report["contrast_score"])

        if quality_report["warnings"]:
            for warning in quality_report["warnings"]:
                st.warning(warning)
        else:
            st.success("Image quality looks good for scanning.")

        boundary_found = document_contour is not None

        if boundary_found:
            document_points = document_contour.reshape(4, 2)
            warped = four_point_transform(image_bgr, document_points)
            boundary_status = "Boundary detected"
        else:
            document_points = None

            if allow_fallback:
                warped = image_bgr.copy()
                boundary_status = "Boundary not found — fallback used"
                st.warning(
                    "Could not find a valid large 4-corner document boundary. "
                    "Fallback mode used: original image was enhanced directly."
                )
            else:
                warped = None
                boundary_status = "Boundary not found"

        if warped is not None:
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

            enhanced_scan = create_enhanced_scan(
                warped,
                enhancement_mode,
                adaptive_block_size,
                adaptive_c
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original")
                st.image(image_np, use_container_width=True)

            with col2:
                st.subheader("Straightened / Source")
                st.image(warped_rgb, use_container_width=True)
                st.caption(boundary_status)

            with col3:
                st.subheader(f"Enhanced: {enhancement_mode}")
                st.image(enhanced_scan, use_container_width=True, clamp=True)

            png_bytes = image_to_png_bytes(enhanced_scan)

            if png_bytes:
                st.download_button(
                    label=f"Download Enhanced Scan - {uploaded_file.name}",
                    data=png_bytes,
                    file_name=f"{base_name}_enhanced_scan.png",
                    mime="image/png"
                )

            extracted_text = ""
            ocr_rows = []
            avg_ocr_confidence = 0.0

            if run_ocr_enabled and reader is not None:
                with st.spinner("Extracting text using OCR..."):
                    extracted_text, ocr_rows, avg_ocr_confidence = run_ocr(
                        reader,
                        enhanced_scan
                    )

                st.subheader("OCR Result")

                ocr_col1, ocr_col2 = st.columns(2)

                with ocr_col1:
                    st.metric("Average OCR Confidence", avg_ocr_confidence)

                with ocr_col2:
                    st.metric("Detected Text Blocks", len(ocr_rows))

                st.text_area(
                    "Extracted Text",
                    extracted_text,
                    height=250,
                    key=f"text_area_{file_index}"
                )

                st.download_button(
                    label=f"Download Extracted Text - {uploaded_file.name}",
                    data=extracted_text,
                    file_name=f"{base_name}_ocr_text.txt",
                    mime="text/plain"
                )

                with st.expander("OCR Confidence Table"):
                    if ocr_rows:
                        st.dataframe(ocr_rows, use_container_width=True)
                    else:
                        st.info("No text detected.")

            scanned_outputs.append(
                {
                    "base_name": base_name,
                    "enhanced_scan": enhanced_scan,
                    "extracted_text": extracted_text,
                    "quality_report": quality_report,
                    "boundary_found": boundary_found,
                    "ocr_rows": ocr_rows,
                    "avg_ocr_confidence": avg_ocr_confidence,
                }
            )

        else:
            st.error(
                "Scanning stopped because no document boundary was found and fallback mode is disabled."
            )

            st.subheader("Original Image")
            st.image(image_np, use_container_width=True)

        # -------------------------
        # Debug Section
        # -------------------------
        if show_debug:
            st.divider()
            st.subheader("Debug Steps")

            st.write("Original image shape:", image_bgr.shape)
            st.write("Grayscale shape:", gray.shape)
            st.write("Boundary status:", boundary_status)
            st.write("Canny lower threshold:", canny_lower)
            st.write("Canny upper threshold:", canny_upper)
            st.write("Blur kernel size:", blur_kernel)
            st.write("Top contours inspected:", top_n_contours)
            st.write("Minimum document area ratio:", min_document_area_ratio)
            st.write("Adaptive threshold block size:", adaptive_block_size)
            st.write("Adaptive threshold constant:", adaptive_c)
            st.write("Enhancement mode:", enhancement_mode)
            st.write("Quality report:", quality_report)

            if candidate_details:
                st.write("Contour candidate details:")
                st.dataframe(candidate_details, use_container_width=True)

            if document_points is not None:
                st.write("Document contour points:", document_points)

            debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(4)

            with debug_col1:
                st.subheader("Grayscale")
                st.image(gray, use_container_width=True, clamp=True)

            with debug_col2:
                st.subheader("Blurred")
                st.image(blurred, use_container_width=True, clamp=True)

            with debug_col3:
                st.subheader("Edges")
                st.image(edges, use_container_width=True, clamp=True)

            with debug_col4:
                st.subheader("Paper Mask")
                st.image(paper_mask, use_container_width=True, clamp=True)

            if contours:
                contour_preview = image_bgr.copy()
                cv2.drawContours(contour_preview, contours, -1, (0, 255, 0), 3)
                contour_preview_rgb = cv2.cvtColor(contour_preview, cv2.COLOR_BGR2RGB)

                st.subheader("Top Contours")
                st.image(contour_preview_rgb, use_container_width=True)
                st.write("Number of top contours shown:", len(contours))

            if document_contour is not None:
                corner_preview = image_bgr.copy()
                cv2.drawContours(corner_preview, [document_contour], -1, (0, 0, 255), 5)
                corner_preview_rgb = cv2.cvtColor(corner_preview, cv2.COLOR_BGR2RGB)

                st.subheader("Selected 4-Corner Boundary")
                st.image(corner_preview_rgb, use_container_width=True)

    # -------------------------
    # Batch Export Section
    # -------------------------
    if scanned_outputs:
        st.divider()
        st.header("Batch Export")

        combined_text = ""

        for idx, item in enumerate(scanned_outputs, start=1):
            combined_text += f"\n\n===== Document {idx}: {item['base_name']} =====\n"
            combined_text += item["extracted_text"] if item["extracted_text"] else "[No OCR text extracted]"

        st.download_button(
            label="Download Combined OCR Text",
            data=combined_text.strip(),
            file_name="combined_ocr_text.txt",
            mime="text/plain"
        )

        zip_buffer = create_zip_file(scanned_outputs)

        st.download_button(
            label="Download All Results as ZIP",
            data=zip_buffer,
            file_name="document_scanner_results.zip",
            mime="application/zip"
        )

        pdf_buffer = create_pdf_file(scanned_outputs)

        if pdf_buffer:
            st.download_button(
                label="Export Enhanced Scans as PDF",
                data=pdf_buffer,
                file_name="enhanced_scans.pdf",
                mime="application/pdf"
            )

else:
    st.info("Upload one or more document images to start scanning.")