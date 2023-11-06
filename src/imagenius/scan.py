from pathlib import Path

import torch

from io import BytesIO

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


gallery_path = Path(__file__).parent.parent.parent / "gallery"

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def scan():
  for image_file in gallery_path.glob("*.jpg"):
    image = Image.open(image_file).convert("RGB")
    # byte_io = BytesIO()
    # image.save(byte_io, "JPEG")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence)[0]


if __name__ == "__main__":
  scan()
