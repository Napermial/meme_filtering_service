from distutils.log import debug
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pytesseract
import numpy as np
import cv2, pillowfight
from PIL import Image
import click
import enchant
import os


def ocr_vanilla_image(image, config) -> str:
    click.echo("running vanilla ocr")
    return pytesseract.image_to_string(image, config=config)


def _create_grayscale_image(image):
    kernel_size = 80
    max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    localMax = cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, max_kernel, None, None, 1, cv2.BORDER_REFLECT101
    )
    gain_division = np.where(localMax == 0, 0, (image / localMax))
    gain_division = np.clip((255 * gain_division), 0, 255)
    gain_division = gain_division.astype("uint8")
    return cv2.cvtColor(gain_division, cv2.COLOR_BGR2GRAY)


def ocr_inverted_colour_image(image, config) -> str:
    click.echo("running inverted colour ocr")
    grayscale_image = _create_grayscale_image(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [3, 3])
    opening = cv2.morphologyEx(grayscale_image, cv2.MORPH_OPEN, kernel)

    _, inverse = cv2.threshold(opening, 200, 255, cv2.THRESH_BINARY_INV)

    return pytesseract.image_to_string(inverse, config=config)


def ocr_grayscale_image_with_otsu(image, config) -> str:
    click.echo("running grayscale with otsu ocr")
    grayscale_image = _create_grayscale_image(image)
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return pytesseract.image_to_string(binary_image, config=config)


def ocr_adaptive_gaussian_treshold_image(image, config) -> str:
    click.echo("running Gaussian adaptive treshold ocr")
    grayscale_image = _create_grayscale_image(image)
    gaussian_image = cv2.adaptiveThreshold(
        grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -15
    )
    _, gaussian_inverted = cv2.threshold(
        gaussian_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return pytesseract.image_to_string(gaussian_inverted, config=config)


def ocr_stroke_width_transformed_image(image, config) -> str:
    click.echo("running stroke width transformed ocr")
    stroke_width_transformed = pillowfight.swt(
        Image.fromarray(image), output_type=pillowfight.SWT_OUTPUT_BW_TEXT
    )
    return pytesseract.image_to_string(stroke_width_transformed, config=config)


def _split_long_string(long: str, dictionary) -> list[str]:
    length = len(long)
    words = set()
    longest = 0
    for i in range(0, length - 2):
        if i < longest:
            continue
        for j in range(length, i + 1, -1):
            if dictionary.check(long[i:j]):
                words.add(long[i:j])
                longest = j
                break
    return words


def merge_solutions(texts: list):
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    click.echo("running merging ocr texts")
    possibles = []
    for text in texts:
        possibles.append(text.replace("\n", " "))
    valid_words = set()
    english_stopwords = set(stopwords.words("english"))
    dictionary = enchant.Dict("en_US")
    for text in possibles:
        for word in text.split(" "):
            word = word.lower()
            if len(word) > 10:
                valid_words.update(
                    [
                        word
                        for word in _split_long_string(word, dictionary)
                        if len(word) > 2 and word.lower() not in english_stopwords
                    ]
                )
                continue
            if (
                len(word) > 2
                and word not in english_stopwords
                and dictionary.check(word)
            ):
                valid_words.add(word)
    click.echo("wordninja finished")
    synonyms = set()
    for word in valid_words:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas()[:1]:
                if len(l.name()) > 2 and l.name().lower() not in english_stopwords:
                    synonyms.add(l.name().lower())
    valid_words.update(synonyms)
    click.echo("synonyms found")
    return valid_words


def load_image_for_ocr(path: str) -> set[str]:
    picture = cv2.imread(path)
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = (
            "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )
    config = "--oem 3 --psm 11 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    try:
        resuts = [
            ocr_vanilla_image(picture, config),
            ocr_inverted_colour_image(picture, config),
            ocr_grayscale_image_with_otsu(picture, config),
            ocr_adaptive_gaussian_treshold_image(picture, config),
            ocr_stroke_width_transformed_image(picture, config),
        ]
    except TypeError:
        return set()
    return merge_solutions(resuts)


if __name__ == "__main__":
    pass
