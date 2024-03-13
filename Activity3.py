import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Images/img1.jpg")
image2 = cv2.imread("Images/img2.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

b, g, r = cv2.split(image)
b2, g2, r2 = cv2.split(image2)

hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_gray2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

hist_r2 = cv2.calcHist([r2], [0], None, [256], [0, 256])
hist_g2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
hist_b2 = cv2.calcHist([b2], [0], None, [256], [0, 256])

fig, axs = plt.subplots(2, 4, figsize=(12, 8))

axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original Image")

axs[0, 1].imshow(gray_image, cmap='gray')
axs[0, 1].set_title("Grayscale Image")

axs[0, 2].plot(hist_gray)
axs[0, 2].set_title("Histogram of Grayscale Image")
axs[0, 2].set_xlabel("Pixel Value")
axs[0, 2].set_ylabel("Frequency")

axs[0, 3].plot(hist_r, color='red')
axs[0, 3].plot(hist_g, color='green')
axs[0, 3].plot(hist_b, color='blue')
axs[0, 3].set_title("RGB Histogram")
axs[0, 3].set_xlabel("Pixel Value")
axs[0, 3].set_ylabel("Frequency")
axs[0, 3].legend(['Red', 'Green', 'Blue'])

axs[1, 0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Original Image 2")

axs[1, 1].imshow(gray_image2, cmap='gray')
axs[1, 1].set_title("Grayscale Image 2")

axs[1, 2].plot(hist_gray2)
axs[1, 2].set_title("Histogram of Grayscale Image 2")
axs[1, 2].set_xlabel("Pixel Value")
axs[1, 2].set_ylabel("Frequency")

axs[1, 3].plot(hist_r2, color='red')
axs[1, 3].plot(hist_g2, color='green')
axs[1, 3].plot(hist_b2, color='blue')
axs[1, 3].set_title("RGB Histogram 2")
axs[1, 3].set_xlabel("Pixel Value")
axs[1, 3].set_ylabel("Frequency")
axs[1, 3].legend(['Red', 'Green', 'Blue'])

plt.tight_layout()
plt.show()