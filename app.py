# === app.py ===
from flask import Flask, request, render_template, redirect, url_for, session
from google.cloud import storage
from google.auth import default
from google.auth.transport.requests import Request
from PIL import Image
import numpy as np
import requests
import os
import tempfile
import uuid

# Konfigurasi dasar
app = Flask(__name__)
app.secret_key = "secretkey"

# GCS config
BUCKET_NAME = "pneumonia-xray-bucket"

# Endpoint Vertex AI
VERTEX_ENDPOINT = "https://asia-southeast2-aiplatform.googleapis.com/v1/projects/mlpt-cloudteam-migration/locations/asia-southeast2/endpoints/320059038752571392:predict"

# Ukuran input ke model
IMG_SIZE = 150

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3).tolist()

def get_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token

def upload_to_gcs(file_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    blob.make_public()  # Agar bisa diakses dari HTML img src
    return blob.public_url

def is_possible_xray(image):
    grayscale_like = sum(
        1 for pixel in image.getdata()
        if abs(pixel[0] - pixel[1]) < 10 and abs(pixel[1] - pixel[2]) < 10
    )
    ratio = grayscale_like / (image.size[0] * image.size[1])
    return ratio > 0.75

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if not file:
            return render_template("index.html", error="Tidak ada file yang dipilih.")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            image = Image.open(tmp_path).convert("RGB")
            if not is_possible_xray(image):
                os.remove(tmp_path)
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
            result = "Pneumonia" if confidence > 0.5 else "Normal"
            reason = "Model mendeteksi indikasi pneumonia." if confidence > 0.5 else "Tidak ditemukan tanda pneumonia."

            # Upload gambar ke GCS
            unique_filename = f"{uuid.uuid4().hex}.jpg"
            public_url = upload_to_gcs(tmp_path, unique_filename)
            os.remove(tmp_path)

            session.update({
                "result": result,
                "reason": reason,
                "image_url": public_url
            })
            return redirect(url_for("hasil"))

        except Exception as e:
            return render_template("index.html", error=f"Terjadi kesalahan: {str(e)}")

    return render_template("index.html")

@app.route("/hasil", methods=["GET", "POST"])
def hasil():
    if request.method == "POST":
        session["doctor_validation"] = request.form.get("validasi")
        return redirect(url_for("validasi"))

    return render_template("hasil.html",
        result=session.get("result"),
        reason=session.get("reason"),
        image_url=session.get("image_url")
    )

@app.route("/validasi")
def validasi():
    return render_template("validasi.html",
        result=session.get("result"),
        reason=session.get("reason"),
        doctor_validation=session.get("doctor_validation"),
        image_url=session.get("image_url")
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
