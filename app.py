# === app.py ===
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from PIL import Image
import numpy as np
import requests
import os

# Konfigurasi
app = Flask(__name__)
app.secret_key = "secretkey"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "sa-vertex.json")
VERTEX_ENDPOINT = "https://asia-southeast2-aiplatform.googleapis.com/v1/projects/mlpt-cloudteam-migration/locations/asia-southeast2/endpoints/320059038752571392:predict"

# Utilitas
IMG_SIZE = 150

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3).tolist()

def get_token():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

def is_possible_xray(image):
    grayscale_like = sum(
        1 for pixel in image.getdata()
        if abs(pixel[0] - pixel[1]) < 10 and abs(pixel[1] - pixel[2]) < 10
    )
    ratio = grayscale_like / (image.size[0] * image.size[1])
    return ratio > 0.75

# Routing
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert("RGB")
            if not is_possible_xray(image):
                os.remove(filepath)
                return render_template("index.html", error="Gambar tidak terdeteksi sebagai rontgen. Silakan upload ulang.")

            instance = preprocess_image(image)
            headers = {
                "Authorization": f"Bearer {get_token()}",
                "Content-Type": "application/json"
            }
            data = {"instances": instance}
            response = requests.post(VERTEX_ENDPOINT, headers=headers, json=data)
            prediction = response.json()
            confidence = prediction['predictions'][0][0]

            if confidence > 0.5:
                result = "Pneumonia"
                reason = "Model mendeteksi indikasi pneumonia. Harap konsultasi ke dokter."
            else:
                result = "Normal"
                reason = "Tidak ditemukan tanda pneumonia pada gambar."

            session.update({
                "result": result,
                "reason": reason,
                "uploaded_file": filename
            })
            return redirect(url_for("hasil"))

        except Exception as e:
            os.remove(filepath)
            return render_template("index.html", error="Terjadi kesalahan dalam memproses gambar. Pastikan gambar valid.")

    return render_template("index.html")

@app.route("/hasil", methods=["GET", "POST"])
def hasil():
    if request.method == "POST":
        session["doctor_validation"] = request.form.get("validasi")
        return redirect(url_for("validasi"))

    return render_template("hasil.html",
        result=session.get("result"),
        reason=session.get("reason"),
        image_url=url_for("static", filename="uploads/" + session.get("uploaded_file", ""))
    )

@app.route("/validasi", methods=["POST"])
def validasi():
    doctor_validation = request.form.get("validasi")
    session["doctor_validation"] = doctor_validation
    return render_template(
        "validasi.html",
        result=session.get("result"),
        reason=session.get("reason"),
        doctor_validation=doctor_validation,
        image_url=url_for("static", filename="uploads/" + session.get("uploaded_file", ""))
    )

if __name__ == "__main__":
    app.run(debug=True)
