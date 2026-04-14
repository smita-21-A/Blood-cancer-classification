from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import sqlite3
import os

app = Flask(__name__)

# ---------------------- LOAD MODEL ----------------------
model = keras.models.load_model("bestmodel.keras")

CLASS_NAMES = [
    "Malignant Early Pre-B",
    "Malignant Pre-B",
    "Malignant Pro-B",
    "Normal"
]


# ---------------------- CONTACT DATABASE FUNCTION ----------------------
def insert_contact(name, email, message):
    conn = sqlite3.connect("contact.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)

    cur.execute(
        "INSERT INTO contact_messages (name, email, message) VALUES (?, ?, ?)",
        (name, email, message)
    )

    conn.commit()
    conn.close()


# ---------------------- FEEDBACK DATABASE FUNCTION ----------------------
def insert_feedback(name, rating, comment):
    conn = sqlite3.connect("bloodshield.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT NOT NULL
        )
    """)

    cur.execute(
        "INSERT INTO feedback_data (name, rating, comment) VALUES (?, ?, ?)",
        (name, rating, comment)
    )

    conn.commit()
    conn.close()


# ---------------------- ROUTES --------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/treatment")
def treatment():
    return render_template("treatment.html")


# ---------- CONTACT FORM ----------
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        insert_contact(name, email, message)

        return render_template("contact.html", success=True)

    return render_template("contact.html")


# ---------- FEEDBACK FORM ----------
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name")
        rating = request.form.get("rating")
        comment = request.form.get("comment")

        insert_feedback(name, rating, comment)

        return render_template("feedback.html", success=True)

    return render_template("feedback.html")


# ---------- PREDICTION ROUTE ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("detect.html", prediction_text="No file uploaded.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("detect.html", prediction_text="No file selected.")

    filepath = "temp.jpg"
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    if os.path.exists(filepath):
        os.remove(filepath)

    return render_template(
        "detect.html",
        prediction_text=f"Prediction: {predicted_class}"
    )


# ---------------------- RUN APP ----------------------
if __name__ == "__main__":
    app.run(debug=True)


