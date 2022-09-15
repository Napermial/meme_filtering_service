import os
from PIL import Image, UnidentifiedImageError
import imagehash


def get_unique_phash_for_images_in_directory(path) -> dict[str, dict[str, str]]:
    picture_metadata = {}

    for root, _, files in os.walk(path):
        for name in files:
            if "webm" not in name and os.path.getsize(os.path.join(root, name)) > 0:
                try:
                    current = Image.open(os.path.join(root, name)).convert("RGB")
                    picture_metadata[str(imagehash.phash(current))] = {
                        "path": os.path.join(root, name)
                    }
                except UnidentifiedImageError:
                    continue

    return picture_metadata


if __name__ == "__main__":
    pass
