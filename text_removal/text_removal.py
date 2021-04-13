import sys
from PIL import Image
import requests
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def removeText(img, threshold=254, close_kernel_size=5, dilate_kernel_size=12):
    """
    Remove the WHITE texts of an image by inpainting

    INPUTS:
        img - PIL.Image.Image or np.ndarray or str: The path of or the PIL 
            Image object of the original image
        threshold - int: the threshold pixel value at above which the pixel 
            will be inpainted
        close_kernel_size - int: the kernel size of the closing operation
        dilate_kernel_size - int: the kernel size of the dilation operation

    RETURNS:
        img_inpainted - PIL.Image.Image: image with text removed/inpainted
    """

    # load image into PIL Image type
    img = loadImage(img)

    # convert image to np arrays
    img_array_gray = np.array(img.convert("L"))
    img_array_colored = np.array(img)

    # generate text mask
    _, mask = cv.threshold(img_array_gray, threshold, 255, cv.THRESH_BINARY)

    kernel_close = cv.getStructuringElement(
        cv.MORPH_RECT, (close_kernel_size, close_kernel_size))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)

    kernel_dilate = cv.getStructuringElement(
        cv.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    mask = cv.dilate(mask, kernel_dilate)

    # inpaint the text regions
    img_inpainted = cv.inpaint(img_array_colored, mask, 15, cv.INPAINT_NS)

    # convert to PIL format
    img_inpainted = Image.fromarray(img_inpainted)

    return img_inpainted


def loadImage(img):
    if isinstance(img, Image.Image):
        pass
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, str):
        if img.startswith("https"):
            img = Image.open(requests.get(img, stream=True).raw)
        else:
            img = Image.open(img)
    else:
        sys.exit("ERROR: Unsupported img type. Abort")

    return img


if __name__ == "__main__":
    img_path = "https://drivendata-public-assets.s3.amazonaws.com/memes-overview.png"
    img = removeText(img_path, threshold=254)
    img.show()
