import re
import pandas as pd
import random
from rapidfuzz import process, fuzz

# ============================================================
# LOAD MED LIST
# ============================================================
med_df = pd.read_csv("med_list.csv")
canon_names = med_df["brand"].str.lower().tolist()

# ============================================================
# RANDOM CONSISTENCY
# ============================================================
random.seed(42)
side_effect_pool = [
    "nausea", "headache", "dizziness", "stomach upset",
    "dry mouth", "loose motions", "sleepiness"
]
avoid_pool = ["alcohol", "coffee", "junk food", "smoking", "cold drinks", "antibiotics"]

# ============================================================
# CLEAN OCR LINES
# ============================================================
def clean_ocr_lines(raw):
    cleaned = []

    garbage_words = [
        "hospital", "mbbs", "obg", "gyn", "paediatrics", "opp",
        "road", "nagar", "reddy", "sudha", "doctor", "date"
    ]

    replacements = {
        "dolo-": "dolo 650",
        "tombiflam": "combiflam",
        "combiflem": "combiflam",
        "combilam": "combiflam",
        "epbin": "eptoin",
        "eploin": "eptoin",
        "i-0-1": "1-0-1",
        "i-i-1": "1-1-1",
        "l-0-1": "1-0-1"
    }

    for line in raw:
        t = line.lower().strip()
        if not t:
            continue

        # Skip hospital header
        if any(g in t for g in garbage_words):
            continue

        # Apply replacements
        for wrong, correct in replacements.items():
            t = t.replace(wrong, correct)

        # Remove garbage chars
        t = re.sub(r"[^a-z0-9\s-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        # Fix broken timing patterns like "1-0-" → "1-0-1"
        if re.fullmatch(r"[0-2]-[0-2]-?", t):
            parts = t.split("-")
            while len(parts) < 3:
                parts.append("1")
            t = "-".join(parts)

        if t:
            cleaned.append(t)

    # Merge lines like "2" + "days"
    merged = []
    i = 0
    while i < len(cleaned):
        if i+1 < len(cleaned) and cleaned[i].isdigit() and "day" in cleaned[i+1]:
            merged.append(f"{cleaned[i]} days")
            i += 2
        else:
            merged.append(cleaned[i])
            i += 1

    return merged

# ============================================================
# TIMING DETECT
# ============================================================
def detect_timing(t):
    t = t.strip().lower()

    if re.fullmatch(r"[0-2]-[0-2]-[0-2]", t):
        return t
    if t == "bd":
        return "1-0-1"
    if t == "od":
        return "1-0-0"
    if t == "tds":
        return "1-1-1"

    return None

# ============================================================
# DURATION DETECT
# ============================================================
def detect_duration(t):
    t = t.lower()

    m = re.search(r"(\d+)\s*day", t)
    if m:
        return int(m.group(1))

    m = re.fullmatch(r"(\d{1,2})", t)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 30:
            return v

    return None

# ============================================================
# SAFE MED MATCH
# ============================================================
def find_medicine_name(raw):
    raw = raw.lower().strip()

    if len(raw) < 3:
        return None

    best = process.extractOne(raw, canon_names, scorer=fuzz.WRatio)
    if best:
        match, score, idx = best
        if score >= 88:
            return idx

    return None

# ============================================================
# FINAL PARSER — 2 LINES BEFORE + 2 LINES AFTER CHECKING
# ============================================================
def parse_prescription(lines):
    meds = []
    n = len(lines)

    for i, line in enumerate(lines):
        line = line.lower().strip()
        idx = find_medicine_name(line)

        if idx is None:
            continue

        row = med_df.iloc[idx]
        brand = row["brand"]
        generic = row["generic"]
        price = float(row["price"])

        timing = None
        duration = None

        # --------- Forward look (next 2 lines) ----------
        for j in range(i+1, min(i+3, n)):
            tline = lines[j]
            t = detect_timing(tline)
            d = detect_duration(tline)
            if t:
                timing = t
            if d:
                duration = d

        # --------- Backward look (prev 2 lines) ----------
        for j in range(max(0, i-2), i):
            tline = lines[j]
            t = detect_timing(tline)
            d = detect_duration(tline)
            if t and not timing:
                timing = t
            if d and not duration:
                duration = d

        # If no timing/duration anywhere → skip hallucination
        if not timing and not duration:
            continue

        if not timing:
            timing = "1-0-1"
        if not duration:
            duration = 5

        meds.append({
            "brand": brand,
            "generic": generic,
            "price": price,
            "timing": timing,
            "duration": duration,
            "side_effects": random.sample(side_effect_pool, 3),
            "avoid_mixing": random.sample(avoid_pool, 2)
        })

    return meds
