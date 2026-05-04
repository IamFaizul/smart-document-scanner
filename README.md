# Smart Document Scanner Pro + OCR

A local Computer Vision application that turns angled document photos into clean scanner-style outputs and extracts text using OCR.

This project was built to practice intermediate Computer Vision concepts using OpenCV, EasyOCR, Streamlit, and FastAPI.

## What It Does

- Upload one or multiple document images
- Detect the document boundary automatically
- Correct perspective and straighten tilted pages
- Enhance the scanned output
- Extract text using OCR
- Show OCR confidence scores
- Analyze image quality before scanning
- Export results as PNG, TXT, ZIP, or PDF
- Includes a FastAPI backend version for API-based usage

## Demo Workflow

Upload document photo → Analyze image quality → Detect page boundary → Fix perspective → Enhance scan → Extract OCR text → Export results

## Key Features

### Document Boundary Detection

The app uses edge detection, contour detection, area filtering, and fallback logic to detect the document region.

It handles cases where smaller rectangles inside a document, such as invoice tables or rows, could be mistakenly selected as the page boundary.

### Perspective Correction

After detecting the four document corners, the app applies a perspective transform to convert the angled image into a straight, top-down scanned view.

### Enhancement Modes

Available enhancement modes:

- Black & White Scan
- Grayscale Scan
- Contrast Enhanced
- Sharpened
- Denoised + Sharpened

### Image Quality Analyzer

The app calculates blur score, brightness score, contrast score, and overall image quality score.

This helps estimate whether the uploaded image is suitable for scanning and OCR.

### OCR Extraction

EasyOCR is used to extract text from the processed scan.

The app also displays extracted text, detected text blocks, average OCR confidence, and confidence score per text block.

### Debug Mode

Debug mode shows the internal CV pipeline:

- Grayscale image
- Blurred image
- Canny edges
- Paper mask
- Top contours
- Selected document boundary
- Contour candidate details

This makes the project useful not only as an app, but also as a Computer Vision learning tool.

## Tech Stack

- Python
- OpenCV
- EasyOCR
- Streamlit
- FastAPI
- NumPy
- Pillow

## Computer Vision Concepts Used

- RGB/BGR conversion
- Grayscale conversion
- Gaussian blur
- Canny edge detection
- Otsu thresholding
- Morphological operations
- Contour detection
- Polygon approximation
- Area-based contour filtering
- Minimum-area rectangle fallback
- Perspective transform
- Adaptive thresholding
- CLAHE contrast enhancement
- Sharpening filters
- OCR confidence scoring

## Project Structure

smart-document-scanner/
├── app.py
├── scanner.py
├── requirements.txt
├── README.md
└── backend/
    ├── main.py
    ├── requirements.txt
    └── services/
        ├── scanner_service.py
        └── ocr_service.py

## Run Streamlit App

pip install -r requirements.txt

streamlit run app.py

## Run FastAPI Backend

cd backend

pip install -r requirements.txt

uvicorn main:app --reload --port 8000

API docs:

http://127.0.0.1:8000/docs

## API Endpoint

POST /scan

Upload an image file and receive:

- Boundary detection status
- Image quality report
- Enhanced scan as base64
- Extracted OCR text
- OCR confidence scores

## Example Response

{
  "file_name": "invoice.png",
  "boundary_found": true,
  "quality_report": {
    "quality_score": 100,
    "status": "Good",
    "blur_score": 2974.05,
    "contrast_score": 73.23
  },
  "ocr": {
    "text": "INVOICE...",
    "average_confidence": 0.91
  }
}

## What I Learned

This project helped me understand how a real Computer Vision pipeline is built step by step:

1. Preprocess the image
2. Detect edges
3. Find contours
4. Filter wrong candidates
5. Detect document corners
6. Apply perspective correction
7. Enhance the result
8. Extract text using OCR
9. Package the logic as both a web app and API

One important challenge was that invoice tables and rows were sometimes detected as document boundaries. I solved this by adding area-based contour filtering, a threshold-based paper mask strategy, and a minimum-area rectangle fallback.

## Status

Completed as a local intermediate-level Computer Vision project.