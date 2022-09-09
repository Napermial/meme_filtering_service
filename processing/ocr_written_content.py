import pyocr
import pytesseract
from cv2 import imread


def ocr_vanilla_image(image, tool) -> str:
    pass


def ocr_inverted_colour_image(image, tool) -> str:
    pass


def ocr_adaptive_gaussian_treshold_image(image, tool) -> str:
    pass


def ocr_stroke_width_transformed_image(image, tool) -> str:
    pass


def merge_solutions():
    pass


def load_image_for_ocr(path: str):
    pyocr.tesseract.TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    tools = pyocr.get_available_tools()
    tool = tools[0]
    picture = imread(path)
    resuts = list(
        ocr_vanilla_image(picture, tool),
        ocr_inverted_colour_image(picture, tool),
        ocr_adaptive_gaussian_treshold_image(picture, tool),
        ocr_stroke_width_transformed_image(picture, tool),
    )
    merge_solutions(resuts)



if __name__ == "__main__":
    pass
