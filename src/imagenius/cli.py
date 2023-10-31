from pathlib import Path

import torch
import requests

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


__name__ = 'imagenius'


app = Flask(__name__, template_folder="./www")
CORS(app) # https://flask-cors.readthedocs.io/en/latest/

MODEL_PORT = 5000

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image_file = Path(__file__).parent.parent.parent / "gallery/street.jpg"
image = Image.open(image_file)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def detect():
    confidence = 0.8

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence)[0]

    client_results = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        client_results.append(
            {
                "label": model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": box,
            }
        )
        # print(
        #         f"Detected {model.config.id2label[label.item()]} with confidence "
        #         f"{round(score.item(), 3)} at location {box}"
        # )

    print(client_results)

    return client_results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("query")

    results = detect()
    print(results)
    # detection = detect(query)

    return jsonify(results)


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
