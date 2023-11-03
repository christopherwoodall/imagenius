from pathlib import Path

import torch
import requests

from io import BytesIO

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


__name__ = 'imagenius'


app = Flask(__name__, template_folder="./www", static_folder="./www/assets")
CORS(app) # https://flask-cors.readthedocs.io/en/latest/

MODEL_PORT = 5000


gallery_path = Path(__file__).parent.parent.parent / "gallery"
# image = Image.open(image_file)


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def detect(query=None):
    confidence = 0.8
    client_results = {}

    for image_file in gallery_path.glob("*.jpg"):
        image = Image.open(image_file).convert("RGB")
        byte_io = BytesIO()
        image.save(byte_io, "JPEG")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if query and model.config.id2label[label.item()] != query:
                continue

            box = [round(i, 2) for i in box.tolist()]

            client_results[f"/gallery/{image_file.name}"] = {
                "label": model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": box,
                "bytes": byte_io.getvalue().hex(),
            }

    return client_results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("query")
    results = detect(query)
    return jsonify(results)


@app.route("/image/<path:filename>")
def image(filename):
    image_data = detect().get(f"/gallery/{filename}")["bytes"]
    byte_io = BytesIO(bytes.fromhex(image_data))
    return send_file(byte_io, mimetype="image/jpeg")


# @app.route("/upload", methods=["POST"])
# def upload_file():
#     file = request.files["file"]
#     # Add logic to send this file to ML Model Server
#     response = requests.post("http://localhost:MODEL_PORT/predict", files={"file": file})
#     tags = response.json()
#     return jsonify({"tags": tags})


def main():
    app.run(
        host="0.0.0.0",
        port=6000,
        ssl_context="adhoc",
        debug=True,
    )


if __name__ == "__main__":
    main()
