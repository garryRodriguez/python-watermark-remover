import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CLEAN_FOLDER = "cleaned"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CLEAN_FOLDER"] = CLEAN_FOLDER

# Create a folder if it is not existing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEAN_FOLDER, exist_ok=True)


def remove_watermark_aggressive(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to YCrCb for luminance detection
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Step 1: Detect subtle watermark regions
    blur = cv2.GaussianBlur(y, (15, 15), 0)
    diff = cv2.absdiff(y, blur)

    # Threshold the difference to generate mask
    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # Change mask to cover watermark edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Step 2: Inpaint to remove watermark
    cleaned = cv2.inpaint(img, mask.astype(np.uint8), 3, cv2.INPAINT_NS)

    return cleaned


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_images():
    uploaded_files = request.files.getlist("images")
    results = []

    for file in uploaded_files:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        cleaned_img = remove_watermark_aggressive(file_path)
        if cleaned_img is None:
            continue

        output_path = os.path.join(app.config["CLEAN_FOLDER"], "cleaned_" + file.filename)
        cv2.imwrite(output_path, cleaned_img)
        results.append(output_path)

    # Show download links
    links = "<h3>Cleaned images:</h3>"
    for img in results:
        filename = os.path.basename(img)
        links += f'<a href="/download/{filename}">{filename}</a><br>'

    return links


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["CLEAN_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)