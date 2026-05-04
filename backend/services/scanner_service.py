import cv2
import numpy as np


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

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


def find_document_contour(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 75, 200)

    edge_contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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

    all_contours = list(edge_contours) + list(paper_contours)
    contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:20]

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area_ratio = 0.20

    for contour in contours:
        area_ratio = cv2.contourArea(contour) / image_area

        if area_ratio < min_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype("int32")

        if cv2.contourArea(box) / image_area >= min_area_ratio:
            return box.reshape(4, 1, 2)

    return None


def enhance_scan(warped_bgr):
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    enhanced = cv2.adaptiveThreshold(
        warped_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return enhanced


def process_document(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quality_report = analyze_image_quality(gray)

    contour = find_document_contour(image_bgr)

    if contour is not None:
        points = contour.reshape(4, 2)
        warped = four_point_transform(image_bgr, points)
        boundary_found = True
    else:
        warped = image_bgr.copy()
        boundary_found = False

    enhanced = enhance_scan(warped)

    return {
        "enhanced": enhanced,
        "quality_report": quality_report,
        "boundary_found": boundary_found,
    }