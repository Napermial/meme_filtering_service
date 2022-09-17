from transformers import (
    ViTFeatureExtractor,
    ViTForImageClassification,
    DetrFeatureExtractor,
    DetrForObjectDetection,
)
from PIL import Image
import torch


def image_classification(image) -> str:
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224"
    )
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def detect_objects(image) -> set[str]:
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
    labels = set()
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        if score > 0.9:
            label_value = model.config.id2label[label.item()]
            labels.update(label_value.replace(',', '').lower().split(' '))
    return labels


def load_image_for_object_detecton(path: str) -> set[str]:
    image = Image.open(path).convert("RGB")
    labels: set[str] = detect_objects(image)
    image = image.resize((224, 224))
    labels.update(image_classification(image).replace(',', '').lower().split(' '))

    return labels


if __name__ == "__main__":
    pass