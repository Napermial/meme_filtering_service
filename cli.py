import click

from preprocessing.deduplicate_images import get_unique_phash_for_images_in_directory
from processing.face_emotion_recognition import predict_emotion
from processing.object_identification import load_image_for_object_detecton
from processing.ocr_written_content import load_image_for_ocr
from processing.face_emotion_recognition import Emotic


@click.command()
@click.option("--input", help="Directory of memes")
@click.option("--output", default="./", help="Where to store the metadata")
@click.option(
    "--only-hashing", default=False, help="Only hashes, deduplicates the images"
)
@click.option("--ocr", default=True, help="Does ocr of the images")
@click.option(
    "--object-category",
    default=True,
    help="Does object detection, image classification on the images",
)
@click.option("--fer", default=True, help="Does face emotion recognition of the images")
def meme_data_enricher(input, output, only_hashing, ocr, object_category, fer):
    meme_metadata = get_unique_phash_for_images_in_directory(input)
    for phash, meta in list(meme_metadata.items()):
        click.echo(f"processing image with hash {phash}")
        tags = set()
        if only_hashing:
            break
        if ocr:
            click.echo("processing ocr")
            ocr_tags = load_image_for_ocr(meta["path"])
            tags.update(ocr_tags)
            meta["ocr"] = ocr_tags
        if object_category:
            click.echo("processing category")
            object_detection_labels = load_image_for_object_detecton(meta["path"])
            tags.update(object_detection_labels)
            meta["object_detection"] = object_detection_labels
        if fer:
            click.echo("processing fer")
            fer_labels = predict_emotion(meta["path"])
            tags.update(fer_labels)
            meta["fer"] = fer_labels
        meta["tags"] = tags
    click.echo(meme_metadata)


if __name__ == "__main__":
    meme_data_enricher()
