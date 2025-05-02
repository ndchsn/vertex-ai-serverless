from flask import Flask, request, render_template, redirect, url_for, session
import requests
import numpy as np
from PIL import Image
import io
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from werkzeug.utils import secure_filename

SERVICE_ACCOUNT_FILE = "/home/andi_jalaluddin/vertex/sa-vertex.json"

app = Flask(__name__)
app.secret_key = "very_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 150, 150, 3).tolist()

def get_token():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None

    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert("RGB")

            # Validasi sederhana: ukuran gambar dan proporsi
            width, height = image.size
            if width > 100 or height < 100 or width > height:
                error_message = "⚠️ Gambar yang diunggah tidak sesuai dengan format foto rontgen. Silakan upload ulang gambar rontgen dada."
                os.remove(filepath)
                return render_template("index.html", error=error_message)

            instance = preprocess_image(image)

            endpoint = "https://asia-southeast2-aiplatform.googleapis.com/v1/projects/mlpt-cloudteam-migration/locations/asia-southeast2/endpoints/320059038752571392:predict"
            headers = {
                "Authorization": f"Bearer {get_token()}",
                "Content-Type": "application/json"
            }
            data = {"instances": instance}
            response = requests.post(endpoint, headers=headers, json=data)
            prediction = response.json()
            confidence = prediction['predictions'][0][0]

            if confidence > 0.5:
                session["result"] = "Pneumonia"
                session["reason"] = """
                    Model mendeteksi tanda-tanda pneumonia berdasarkan analisis gambar rontgen.
                    Pneumonia dapat disebabkan oleh infeksi bakteri seperti *Streptococcus pneumoniae* atau virus seperti influenza dan COVID-19.
                    Tanda-tanda pada rontgen dapat meliputi infiltrat, konsolidasi, atau opasitas lokal.
                    Segera konsultasikan dengan dokter untuk tindakan medis yang tepat.
                """
            else:
                session["result"] = "Normal"
                session["reason"] = """
                    Model tidak menemukan tanda-tanda pneumonia yang mencolok pada gambar rontgen.
                    Paru-paru terlihat bersih dan bebas dari infeksi akut.
                    Namun, tetap lakukan pemeriksaan medis jika terdapat keluhan kesehatan.
                """

            session["uploaded_file"] = filename
            return redirect(url_for("hasil"))

        except Exception:
            os.remove(filepath)
            error_message = "❌ File yang diunggah tidak dapat diproses. Pastikan ini adalah gambar rontgen dengan format yang benar."
            return render_template("index.html", error=error_message)

    return render_template("index.html")


@app.route("/hasil", methods=["GET", "POST"])
def hasil():
    doctor_validation = None
    if request.method == "POST":
        doctor_validation = request.form.get("validasi")
        session["doctor_validation"] = doctor_validation

    return render_template(
        "hasil.html",
        result=session.get("result"),
        reason=session.get("reason"),
        doctor_validation=session.get("doctor_validation"),
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
