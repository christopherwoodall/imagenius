from pathlib import Path

import torch

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


# Path to the gallery directory containing images
gallery_path = Path(__file__).parent.parent.parent / "gallery"

# Initialize the processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Set confidence threshold for object detection
confidence_threshold = 0.8


def scan_and_print_tags():
  for image_file in gallery_path.glob("*.jpg"):
    image = Image.open(image_file).convert("RGB")  # Convert image to RGB

    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process the outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

    # Print out the results
    print(f"Tags for image {image_file.name}:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
      if score >= confidence_threshold:  # Check if the score is above the threshold
        label_name = model.config.id2label[label.item()]
        box_data = [round(b, 2) for b in box.tolist()]
        print(f"\tLabel: {label_name}, Score: {score.item():.2f}, Box: {box_data}")

  print("Scanning complete.")


def main():
    scan_and_print_tags()


if __name__ == "__main__":
  main()
