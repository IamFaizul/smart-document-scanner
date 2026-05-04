import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.scanner_service import process_document
from services.ocr_service import extract_text


app = FastAPI(
    title="Smart Document Scanner API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_upload_as_bgr(file_bytes: bytes):
    np_array = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Invalid image file.")

    return image_bgr


def encode_image_to_base64(image):
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Could not encode processed image.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Smart Document Scanner API is running",
    }


@app.post("/scan")
async def scan_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        image_bgr = read_upload_as_bgr(file_bytes)

        scan_result = process_document(image_bgr)

        enhanced_image = scan_result["enhanced"]
        enhanced_base64 = encode_image_to_base64(enhanced_image)

        ocr_result = extract_text(enhanced_image)

        return {
            "file_name": file.filename,
            "boundary_found": scan_result["boundary_found"],
            "quality_report": scan_result["quality_report"],
            "enhanced_image_base64": enhanced_base64,
            "ocr": ocr_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))