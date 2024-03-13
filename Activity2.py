import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Images/img2.jpg")

height, width, _ = image.shape
print("Image size: {} x {}".format(width, height))

pixel = image[333, 333]
print("Pixel value (R, G, B):", pixel)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray, 50, 150)
edges2 = cv2.Canny(gray, 100, 200)
edges3 = cv2.Canny(gray, 150, 250)
edges4 = cv2.Canny(gray, 200, 300)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(edges1, cmap='gray')
axs[0, 0].set_title('Edges 1')
axs[0, 1].imshow(edges2, cmap='gray')
axs[0, 1].set_title('Edges 2')
axs[1, 0].imshow(edges3, cmap='gray')
axs[1, 0].set_title('Edges 3')
axs[1, 1].imshow(edges4, cmap='gray')
axs[1, 1].set_title('Edges 4')

plt.show()

b, g, r = cv2.split(image)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(b, cmap='gray')
axs[0].set_title('Blue Channel')
axs[1].imshow(g, cmap='gray')
axs[1].set_title('Green Channel')
axs[2].imshow(r, cmap='gray')
axs[2].set_title('Red Channel')

plt.show()