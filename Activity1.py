import cv2
import numpy as np

image = cv2.imread("Images/img1.jpg")

cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

row, col = 333, 333
pixel_value = gray_image[row, col]
print("Pixel value at ({}, {}):".format(row, col), pixel_value)

_, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Black and White Image", bw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()