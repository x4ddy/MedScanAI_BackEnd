import os
import cv2
import easyocr
import numpy as np
import pandas as pd
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

from parser import clean_ocr_lines, parse_prescription

# ---------------------------------------------------
# INITIALIZE APP
# ---------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# LOAD MEDICINE DATABASE
# ---------------------------------------------------
med_df = pd.read_csv("med_list.csv")   # MUST EXIST


# ---------------------------------------------------
# GLOBAL ERROR HANDLER
# ---------------------------------------------------
@app.errorhandler(Exception)
def handle_exception(e):
    print("🔥 SERVER ERROR:", e)
    return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------
def preprocess(img_path):
    print("[1] Preprocessing image...")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image file not found or unreadable")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=25)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 3
    )

    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out_path = "processed.png"
    cv2.imwrite(out_path, thresh)

    return out_path


# ---------------------------------------------------
# OCR STEP
# ---------------------------------------------------
def run_ocr(img_path):
    print("[2] Running OCR...")

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img_path, detail=0)

    print("[DEBUG RAW OCR]:", result)
    return result


# ---------------------------------------------------
# COST SAVING CALCULATION
# ---------------------------------------------------
def calculate_savings(meds):
    random.seed(42)  # deterministic multiplier

    total_generic = 0
    total_brand = 0

    for m in meds:
        generic_price = float(m["price"])
        multiplier = random.uniform(2, 3)
        brand_estimated = generic_price * multiplier

        m["generic_price"] = round(generic_price, 2)
        m["brand_estimated_price"] = round(brand_estimated, 2)
        m["savings"] = round(brand_estimated - generic_price, 2)

        total_generic += generic_price
        total_brand += brand_estimated

    return {
        "total_generic_cost": round(total_generic, 2),
        "total_brand_estimated_cost": round(total_brand, 2),
        "total_savings": round(total_brand - total_generic, 2),
    }


# ---------------------------------------------------
# MAIN API ROUTE — OCR + PARSE + SAVINGS
# ---------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        img_path = "upload_tmp.jpg"
        file.save(img_path)

        processed = preprocess(img_path)
        raw_lines = run_ocr(processed)

        print("[3] Cleaning OCR...")
        cleaned = clean_ocr_lines(raw_lines)
        print("[DEBUG CLEANED]:", cleaned)

        print("[4] Parsing prescription...")
        meds = parse_prescription(cleaned)

        # convert to python vars
        for m in meds:
            m["price"] = float(m["price"])
            m["duration"] = int(m["duration"])

        # apply cost savings
        print("[5] Adding cost savings...")
        savings_info = calculate_savings(meds)

        # build summary
        summary_lines = []
        for m in meds:
            summary_lines.append(
                f"{m['brand']} — take {m['timing']} for {m['duration']} days."
            )
            summary_lines.append(f" • Avoid: {', '.join(m['avoid_mixing'])}")
            summary_lines.append(f" • Side effects: {', '.join(m['side_effects'])}")
            summary_lines.append(
                f" • Generic: ₹{m['generic_price']} | Estimated Branded: ₹{m['brand_estimated_price']} | Save: ₹{m['savings']}"
            )
            summary_lines.append("")

        final_summary = "\n".join(summary_lines).strip()

        # table for UI
        table = [
            {
                "brand": m["brand"],
                "generic": m["generic"],
                "price": m["generic_price"],
                "brand_estimated_price": m["brand_estimated_price"],
                "savings": m["savings"],
                "duration": m["duration"],
                "timing": m["timing"],
            }
            for m in meds
        ]

        return jsonify({
            "summary": final_summary,
            "table": table,
            "savings": savings_info
        })

    except Exception as e:
        print("🔥 ERROR DURING PROCESSING:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# SEARCH API — REVERTED TO YOUR OLD VERSION
# ---------------------------------------------------
@app.route("/search", methods=["GET"])
def search_medicine():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify({"results": []})

    try:
        mask = (
            med_df["brand"].astype(str).str.lower().str.contains(query, na=False) |
            med_df["generic"].astype(str).str.lower().str.contains(query, na=False)
        )

        results = med_df[mask][["brand", "generic", "price"]].to_dict(orient="records")

        return jsonify({"results": results})

    except Exception as e:
        print("🔥 SEARCH ERROR:", e)
        return jsonify({"results": []})


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
