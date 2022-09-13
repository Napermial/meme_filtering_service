import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pytesseract
import numpy as np
import cv2, pillowfight
from PIL import Image
from textblob import TextBlob
import wordninja


def ocr_vanilla_image(image, config) -> str:
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
    grayscale_image = _create_grayscale_image(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [3, 3])
    opening = cv2.morphologyEx(grayscale_image, cv2.MORPH_OPEN, kernel)

    _, inverse = cv2.threshold(opening, 200, 255, cv2.THRESH_BINARY_INV)

    return pytesseract.image_to_string(inverse, config=config)


def ocr_grayscale_image_with_otsu(image, config) -> str:
    grayscale_image = _create_grayscale_image(image)
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return pytesseract.image_to_string(binary_image, config=config)


def ocr_adaptive_gaussian_treshold_image(image, config) -> str:
    grayscale_image = _create_grayscale_image(image)
    gaussian_image = cv2.adaptiveThreshold(
        grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -15
    )
    return pytesseract.image_to_string(gaussian_image, config=config)


def ocr_stroke_width_transformed_image(image, config) -> str:
    stroke_width_transformed = pillowfight.swt(
        Image.fromarray(image), output_type=pillowfight.SWT_OUTPUT_GRAYSCALE_TEXT
    )
    return pytesseract.image_to_string(stroke_width_transformed, config=config)


def merge_solutions(texts: list):
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    possibles = []
    for text in texts:
        textBlb = TextBlob(text.replace("\n", " "))
        possibles.append(textBlb.correct())

    valid_words = set()

    for text in possibles:
        for word in text.split(" "):
            word = word.lower()
            if len(word) > 2 and word not in set(stopwords.words("english")):
                if len(word) > 10:
                    valid_words.update(
                        [word for word in wordninja.split(word) if len(word) > 2]
                    )
                    continue
                valid_words.add(word)

    synonyms = set()
    for word in valid_words:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas()[:1]:
                synonyms.add(l.name().lower())
    valid_words.update(synonyms)
    return valid_words


def load_image_for_ocr(path: str) -> set[str]:
    picture = cv2.imread(path)
    pytesseract.pytesseract.tesseract_cmd = (
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
    config = "--oem 3 --psm 11 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    resuts = [
        ocr_vanilla_image(picture, config),
        ocr_inverted_colour_image(picture, config),
        ocr_grayscale_image_with_otsu(picture, config),
        ocr_adaptive_gaussian_treshold_image(picture, config),
        ocr_stroke_width_transformed_image(picture, config),
    ]
    return merge_solutions(resuts)


if __name__ == "__main__":
    pass
