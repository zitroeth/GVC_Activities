import cv2
import numpy as np
import matplotlib.pyplot as plt

image2 = cv2.imread('Images/image2.jpg')

gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image2, 100, 200)

histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')

axs[0, 1].imshow(edges, cmap='gray')
axs[0, 1].set_title('Edges')

axs[1, 0].imshow(gray_image, cmap='gray')
axs[1, 0].set_title('Grayscale Image')

axs[1, 1].plot(histogram)
axs[1, 1].set_title('Histogram')

print('Filename:', 'img2.jpg')
print('Format:', 'RGB')
print('Width:', image2.shape[1])
print('Height:', image2.shape[0])
print('Size:', image2.size)

pixel_value = image2[100, 100]
print('Pixel value at (100, 100):', pixel_value)

plt.show()