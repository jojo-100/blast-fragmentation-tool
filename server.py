"""
Local server for the Blast Fragmentation Calculator.

It does the two jobs the browser cannot do on its own:
  1. /predict_rock_factor  -> runs your trained ML model (RandomForest) to turn
                              rock-mass properties into rock factor A.
  2. /analyze_photo        -> sends an uploaded rock photo to a vision model on
                              OpenRouter and turns the result into the five
                              property numbers the ML model needs.

The Kuz-Ram fragmentation math still runs in the browser (in the HTML/JS).
This server just adds the ML + vision pieces behind it.

The API key is read from the environment variable OPENROUTER_API_KEY and never
leaves this machine / never reaches the browser.

Files expected in the SAME folder as this script:
  - blast_calculator_server.html   (the page)
  - rock_factor_training_data.csv  (training data for the model)

Run with:   python server.py
Then open:  http://127.0.0.1:5000   (it also opens automatically)
"""

import os
import json
import base64
import threading
import webbrowser

import pandas as pd
from flask import Flask, request, jsonify, send_file
from sklearn.ensemble import RandomForestRegressor
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(HERE, "blast_calculator_server.html")
CSV_FILE  = os.path.join(HERE, "rock_factor_training_data.csv")

# Same free vision model your step-2 script used. If it's busy, OpenRouter's
# auto-router "openrouter/free" will pick another free vision-capable model.
VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"

app = Flask(__name__)

# ---------------------------------------------------------------
# Train the ML model once, at startup (same logic as your training script)
# ---------------------------------------------------------------
FEATURES = ["RMD", "JPS", "JPA", "RDI", "HF"]
TARGET   = "rock_factor_A"

print("Loading training data and training the rock-factor model...")
_data = pd.read_csv(CSV_FILE)
_model = RandomForestRegressor(n_estimators=100, random_state=42)
_model.fit(_data[FEATURES], _data[TARGET])
print(f"  Model trained on {len(_data)} rows. Ready.")

# ---------------------------------------------------------------
# Mappings: turn the vision model's words into the exact numeric ratings
# the ML model was trained on (taken from your training data).
# ---------------------------------------------------------------
JPS_MAP = {"<0.1m": 10, "0.1m_to_oversize": 20, "oversize_to_pattern": 50}
JPA_MAP = {"dip_out_of_face": 20, "strike_perpendicular": 30, "dip_into_face": 40}
# Hardness is the fuzziest thing to read off a photo, so we map a coarse
# category to a representative hardness factor (HF). User can override on the page.
HF_MAP  = {"soft": 5.0, "medium": 15.0, "hard": 30.0, "very_hard": 45.0}


def properties_from_vision(v):
    """Convert the vision model's structured answer into RMD, JPS, JPA, RDI, HF."""
    jps = JPS_MAP.get(v.get("joint_spacing"), 20)          # default: medium
    jpa = JPA_MAP.get(v.get("joint_orientation"), 30)      # default: medium
    jf  = jps + jpa                                         # joint factor

    rmt = v.get("rock_mass_type", "blocky_jointed")
    if rmt == "powdery_friable":
        rmd = 10
    elif rmt == "massive":
        rmd = 50
    else:  # blocky/jointed rock -> RMD equals the joint factor (Cunningham)
        rmd = jf

    density = v.get("density", 2.7)
    try:
        density = float(density)
    except (TypeError, ValueError):
        density = 2.7
    density = max(2.2, min(3.2, density))   # keep within trained range
    rdi = 25.0 * density - 50.0             # same formula as the training data

    hf = HF_MAP.get(v.get("hardness"), 15.0)

    return {
        "RMD": round(rmd, 1),
        "JPS": jps,
        "JPA": jpa,
        "RDI": round(rdi, 2),
        "HF": round(hf, 2),
    }


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    return send_file(HTML_FILE)


@app.route("/predict_rock_factor", methods=["POST"])
def predict_rock_factor():
    """Run the trained model on the five properties and return rock factor A."""
    try:
        body = request.get_json(force=True)
        row = pd.DataFrame([{f: float(body[f]) for f in FEATURES}])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Need numeric RMD, JPS, JPA, RDI, HF."}), 400

    a = float(_model.predict(row)[0])
    return jsonify({"A": a})


@app.route("/analyze_photo", methods=["POST"])
def analyze_photo():
    """Send the uploaded photo to the vision model, return the 5 properties."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENROUTER_API_KEY not set in this terminal. "
                                 "Set it, then restart the server."}), 400

    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded."}), 400

    f = request.files["photo"]
    raw = f.read()
    if not raw:
        return jsonify({"error": "Uploaded photo was empty."}), 400

    mime = "image/png" if f.filename.lower().endswith(".png") else "image/jpeg"
    data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"

    prompt = (
        "You are a mining geologist assessing a rock face/outcrop photo for blast "
        "design. Reply with ONLY a JSON object (no markdown, no extra text) with "
        "EXACTLY these keys:\n"
        '{\n'
        '  "rock_mass_type": "powdery_friable" | "massive" | "blocky_jointed",\n'
        '  "joint_spacing": "<0.1m" | "0.1m_to_oversize" | "oversize_to_pattern",\n'
        '  "joint_orientation": "dip_out_of_face" | "strike_perpendicular" | "dip_into_face",\n'
        '  "density": a number between 2.2 and 3.2 (estimated rock density, t/m3),\n'
        '  "hardness": "soft" | "medium" | "hard" | "very_hard",\n'
        '  "notes": "one short sentence on what is actually visible"\n'
        '}\n'
        "If a property is not visible, choose the most typical value and say so in notes."
    )

    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": VISION_MODEL,
                  "messages": [{"role": "user", "content": [
                      {"type": "text", "text": prompt},
                      {"type": "image_url", "image_url": {"url": data_uri}},
                  ]}]},
            timeout=120,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach OpenRouter: {e}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"OpenRouter returned {resp.status_code}: {resp.text[:300]}"}), 502

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        return jsonify({"error": "Unexpected response from vision model."}), 502

    # The model should return JSON, but free models sometimes wrap it in prose or
    # code fences. Pull out the {...} block and parse it defensively.
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        parsed = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return jsonify({"error": "Vision model did not return clean JSON. "
                                 "Raw output shown; fill the properties by hand.",
                        "raw": content}), 200

    props = properties_from_vision(parsed)
    notes = parsed.get("notes", "")
    description = (
        f"Vision model read:\n"
        f"  rock mass: {parsed.get('rock_mass_type','?')}\n"
        f"  joint spacing: {parsed.get('joint_spacing','?')}\n"
        f"  joint orientation: {parsed.get('joint_orientation','?')}\n"
        f"  density: {parsed.get('density','?')} t/m3\n"
        f"  hardness: {parsed.get('hardness','?')}\n"
        f"  notes: {notes}\n\n"
        f"-> mapped to RMD={props['RMD']}, JPS={props['JPS']}, JPA={props['JPA']}, "
        f"RDI={props['RDI']}, HF={props['HF']}\n"
        f"(estimates from a photo -- edit any value before predicting A.)"
    )

    out = dict(props)
    out["description"] = description
    return jsonify(out)


# ---------------------------------------------------------------
# Start
# ---------------------------------------------------------------
if __name__ == "__main__":
    # PORT is set by the host (Render, Railway, etc.). Locally it falls back to 5000.
    port = int(os.environ.get("PORT", 5000))
    # When running on your own machine (no PORT from a host), open the browser.
    if not os.environ.get("PORT"):
        url = f"http://127.0.0.1:{port}"
        print(f"\nServer running. Open {url} (opening it for you now).")
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("NOTE: OPENROUTER_API_KEY is not set -- the calculator and ML work,\n"
                  "      but photo analysis will error until you set the key and restart.")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    else:
        print(f"Server starting on port {port} (deployed mode).")
    # 0.0.0.0 lets the host route external traffic to the app.
    app.run(host="0.0.0.0", port=port, debug=False)
