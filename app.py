from pathlib import Path
import io
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "medical_image_cnn.keras"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
_model = None

def preprocess_image(raw):
    try:
        image = Image.open(io.BytesIO(raw)).convert("L").resize((256, 256))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Please upload a valid image file.") from exc
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, 256, 256, 1)

def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

@app.route("/", methods=["GET", "POST"])
def index():
    result = error = message = None
    if not MODEL_PATH.exists():
        message = "No trained model found. Add the dataset and run python src/train.py."
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "Choose an image first."
        else:
            try:
                tensor = preprocess_image(uploaded.read())
                model = get_model()
                if model is None:
                    message = "No trained model found. Follow the README to train one."
                else:
                    score = float(np.asarray(model.predict(tensor, verbose=0)).reshape(-1)[0])
                    result = {"label": "PNEUMONIA" if score >= 0.5 else "NORMAL",
                              "score": f"{score:.3f}"}
            except ValueError as exc:
                error = str(exc)
    return render_template("index.html", result=result, error=error, message=message)

@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", result=None, error="Maximum upload size is 8 MB.", message=None), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
