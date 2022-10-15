# Meme filtering tool

This tool allows you to gather valuabe metadata from internet culture images.

- Deduplication
The images are deduplicated with a hashing algorythm [imagehash/perceptual hashing](https://github.com/JohannesBuchner/imagehash) which is used as a primary identificator in the metadata file.
- OCR
First the text from the image is read with tesseract.  The image is prepared with transformations such as Gaussian adaptive tresholding and  
- Object identification
[Facebook's-detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) is used to gather info of the objects in the image
- Image classification
The image is classified to one of [ImageNet's](https://github.com/Alibaba-MIIL/ImageNet21K) 1000 categories using google's [Vision Transformer model](https://huggingface.co/google/vit-base-patch16-224)
- Face emotion recognition
Faces and emotions are recovered from the image with the help of the [Emotic model](https://github.com/Tandon-A/emotic)

## Usage

You can use the cli after installing the dependencies from constraints.txt

0. Download and install [tesseract](https://github.com/tesseract-ocr/tesseract#installing-tesseract) with the English data packages -make sure it is on the path
1. Create a virtual environment to avoid package collision and activate it

```bash
python -m venv ./venv
. venv/bin/activate
```

2. install dependencies

```bash
pip install -r requirements.txt
```

3. You can now run

```bash
meme-data-enricher input=path/to/directory/with/memes
```

## CLI options

- "--input" path of the input directory, required
- "--output" Where to store the metadata, has to be set with filename and extension (json is preferred for easy formatting), optional e.g.: /output.json
- "--only-hashing" Only hashes, deduplicates the images, default=False, optional
- "--ocr" ocr is run on image,  default=True, optional
- "--object-category" Does object detection, image classification on the images, default=True, optional
- "--fer", Does face emotion recognition of the images, default=True, optional

## Metadata shape
```JSON
{
    "perceptual_hash":
    {
        "path":"path/of/image",
        "ocr": ["tag", "gathered", "during", "ocr"],
        "object_detection" : ["tag", "gathered", "during", "object_detection/classification"],
        "fer":["tags", "gathered", "during", "fer"],
        "tags":["all", "tags", "deduplicated"]
    }
}
```
## Key-value metadata shape
```JSON
{
    "tag":
    {
        "images":["perceptual_hash", "perceptual_hash"]
    }
}
```

Future plans:
collect [famous people](https://medialab.github.io/bhht-datascape/) and incorporate a [face recognition algorythm](https://github.com/shobhit9618/celeb_recognition)
