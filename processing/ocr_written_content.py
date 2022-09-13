import pyocr
import pytesseract
import numpy as np
import cv2


def ocr_vanilla_image(image, tool) -> str:
    return tool.image_to_string(image, lang="eng", builder=pyocr.builders.TextBuilder())


def _create_grayscale_image(image):
    kernel_size = 80
    max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    localMax = cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, max_kernel, None, None, 1, cv2.BORDER_REFLECT101
    )
    # Perform gain division
    gain_division = np.where(localMax == 0, 0, (image / localMax))

    # Clip the values to [0,255]
    gain_division = np.clip((255 * gain_division), 0, 255)

    # Convert the mat type from float to uint8:
    gain_division = gain_division.astype("uint8")
    return cv2.cvtColor(gain_division, cv2.COLOR_BGR2GRAY)


def ocr_inverted_colour_image(image, tool) -> str:
    grayscale_image = _create_grayscale_image(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [3, 3])
    opening = cv2.morphologyEx(grayscale_image, cv2.MORPH_OPEN, kernel)

    _, inverse = cv2.threshold(opening, 200, 255, cv2.THRESH_BINARY_INV)

    return tool.image_to_string(
        inverse, lang="eng", builder=pyocr.builders.TextBuilder()
    )


def ocr_grayscale_image_with_otsu(image, tool) -> str:
    grayscale_image = _create_grayscale_image(image)
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return tool.image_to_string(
        binary_image, lang="eng", builder=pyocr.builders.TextBuilder()
    )


def ocr_adaptive_gaussian_treshold_image(image, tool) -> str:
    grayscale_image = _create_grayscale_image(image)

    # mean treshold
    mean_image = cv2.adaptiveThreshold(
        grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    gaussian_image = cv2.adaptiveThreshold(
        grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -15
    )


def ocr_stroke_width_transformed_image(image, tool) -> str:
    pass


def merge_solutions(texts: list):
    pass


def load_image_for_ocr(path: str):
    pyocr.tesseract.TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    tools = pyocr.get_available_tools()
    tool = tools[0]
    picture = cv2.imread(path)
    resuts = list(
        ocr_vanilla_image(picture, tool),
        ocr_inverted_colour_image(picture, tool),
        ocr_grayscale_image_with_otsu(picture, tool),
        ocr_adaptive_gaussian_treshold_image(picture, tool),
        ocr_stroke_width_transformed_image(picture, tool),
    )
    merge_solutions(resuts)


if __name__ == "__main__":
    pass
