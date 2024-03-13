import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Images/img1.jpg")
image2 = cv2.imread("Images/img2.jpg")
original_image = cv2.imread('Images/img2.jpg')

def main():
    while(1):
        option = int(input("Enter Activity: "))
        match option:
            case 1:
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
            case 2:
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
            case 3:
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
            case 4:
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
            case 5:
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
            case _:
                break
main()