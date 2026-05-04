import easyocr


_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def extract_text(image):
    reader = get_reader()
    results = reader.readtext(image)

    rows = []
    lines = []

    for _, text, confidence in results:
        rows.append(
            {
                "text": text,
                "confidence": round(float(confidence), 3),
            }
        )
        lines.append(text)

    extracted_text = "\n".join(lines)

    if rows:
        avg_confidence = round(
            sum(row["confidence"] for row in rows) / len(rows),
            3
        )
    else:
        avg_confidence = 0.0

    return {
        "text": extracted_text,
        "blocks": rows,
        "average_confidence": avg_confidence,
    }