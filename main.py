import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import sys

if __name__ == "__main__":

    if "-filePath" in sys.argv:
        filePath = sys.argv[sys.argv.index("-filePath") + 1]
    else:
        print("No input file")
        sys.exit()

    # Read in Image, Grayscale and Blur
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filter and find edges for localization
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

    # Find Contours and Apply Mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    # Use Easy OCR To Read Text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    # Render Result
    text = result[0][-2]
    print(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

    img.save("result.jpg")
