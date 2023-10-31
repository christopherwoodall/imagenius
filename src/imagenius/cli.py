from flask import Flask, request, jsonify, render_template
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from pathlib import Path


app = Flask(__name__, template_folder="./www")

MODEL_PORT = 5000

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image_file = Path(__file__).parent.parent.parent / "gallery/street.jpg"
image = Image.open(image_file)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def detect():
    config = 0.8

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )


@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files["file"]
#     # Add logic to send this file to ML Model Server
#     response = requests.post("http://localhost:MODEL_PORT/predict", files={"file": file})
#     tags = response.json()
#     return jsonify({"tags": tags})


def main():
    app.run(debug=True, port=6000)
