import json


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def save_metadata_to_file(path: str, metadata):
    with open(path, "w+") as f:
        f.write(json.dumps(metadata, cls=SetEncoder))


if __name__ == "__main__":
    pass
