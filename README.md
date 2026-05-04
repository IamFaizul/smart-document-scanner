# ScanCraft AI: Smart Document Scanner with OCR

A Computer Vision project that scans document images, detects document boundaries, corrects perspective, enhances readability, extracts text using OCR, and exports results in multiple formats.

## Project Overview

This project takes one or more document images as input and converts them into clean scanner-like outputs. It uses OpenCV for image processing and perspective correction, EasyOCR for text extraction, and Streamlit for the interactive web interface.

The project is designed as a practical Computer Vision learning project covering image preprocessing, edge detection, contour detection, perspective transformation, OCR, image quality analysis, and batch export workflows.

## Features

- Upload single or multiple document images
- Detect document boundaries automatically
- Crop and straighten tilted documents
- Enhance scanned output using multiple modes
- Extract text using OCR
- Show OCR confidence scores
- Analyze image quality before scanning
- Tune OpenCV parameters from the sidebar
- Use fallback mode if document boundary detection fails
- View debug steps for learning and troubleshooting
- Download enhanced scanned images
- Download extracted OCR text
- Export all results as ZIP
- Export enhanced scans as a single PDF

## Computer Vision Concepts Used

- RGB to BGR conversion
- Grayscale conversion
- Gaussian blur
- Canny edge detection
- Contour detection
- Polygon approximation
- 4-corner document boundary detection
- Perspective transformation
- Adaptive thresholding
- CLAHE contrast enhancement
- Sharpening filters
- Image quality scoring
- OCR text extraction

## Tech Stack

- Python
- OpenCV
- NumPy
- Streamlit
- EasyOCR
- Pillow

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd smart-document-scanner