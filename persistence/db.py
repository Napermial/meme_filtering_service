import pickledb
import json
from collections import defaultdict
import os


class MetaDataBase:
    images: pickledb.PickleDB
    tags: pickledb.PickleDB

    def __init__(self):
        path = os.environ.get("DB_PATH")
        self.images = pickledb.load(path, True, sig=False)
        with open(path, "r") as f:
            self.tags = self.extract_tags_from_metadata(json.load(f))

    def extract_tags_from_metadata(self, metadata: dict) -> pickledb.PickleDB:
        tags = defaultdict(list)
        for hash, meta in metadata.items():
            for tag in meta["tags"]:
                tags[tag].append(hash)
        tags_path = os.path.join(os.getcwd(), "tags.json")
        with open(tags_path, "w") as f:
            json.dump(tags, f)
        tags_db = pickledb.load(tags_path, True, sig=False)
        return tags_db

    def _get_relevant_tags(self, search: str):
        return [tag for tag in self.tags.getall() if tag.startswith(search)]

    def get_autocomplete_suggestions(self, search: str) -> list[str]:
        return self._get_relevant_tags(search)

    def get_image_suggestions(self, search: str) -> list[str]:
        relevant_tags = self._get_relevant_tags(search)
        hashes = set()
        for tag in relevant_tags:
            hashes.update(self.tags.get(tag))
        images = []
        for hash in hashes:
            images.append(self.images.dget(hash, 'path'))
                
        return images

    def add_image(self, metadata: dict):
        pass


if __name__ == "__main__":
    meme = MetaDataBase(
        "/home/napermia/documents/projektek/meme_filtering_service/metadata.json"
    )

    print(meme.get_autocomplete_suggestions("el"))
