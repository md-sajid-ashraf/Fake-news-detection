import os
import sys

import joblib
from flask import Flask, render_template, request

# Project root (parent of this folder) holds model.pkl and preprocessing.py
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PARENT_DIR)
from preprocessing import wordpre  # noqa: E402  (shared with train_model.py)

app = Flask(__name__)

# File name is case-sensitive on Linux/macOS: it is "model.pkl", not "Model.pkl".
MODEL_PATH = os.path.join(PARENT_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    text = request.form.get("txt", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some news text first.")

    cleaned = wordpre(text)
    if not cleaned.strip():
        return render_template("index.html", text=text,
                               error="The text has no usable words after cleaning. Please enter a real article.")
    try:
        result = int(model.predict([cleaned])[0])  # 1 = real, 0 = fake
    except Exception as exc:  # show the problem instead of mislabelling it as "True News"
        return render_template("index.html", text=text, error=f"Prediction failed: {exc}")

    return render_template("index.html", text=text, result=result)


if __name__ == "__main__":
    # debug mode exposes an interactive debugger; enable only locally: FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
