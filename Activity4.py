import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Images/img2.jpg")

plt.subplot(231)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(232)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title("Grayscale Image")

plt.subplot(233)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title("Grayscale Histogram")

edges = cv2.Canny(gray_image, 100, 200)
plt.subplot(234)
plt.imshow(edges, cmap='gray')
plt.title("Edges")

plt.tight_layout()
plt.show()