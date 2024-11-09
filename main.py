import cv2
from Clustering_filter import Clustering_Filter 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = cv2.imread("a.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    pre_image = Clustering_Filter(gray_image, height, width, 0.5, 5, 40, 0.5)
    enhancement_image = pre_image.process_image()
    plt.imshow(enhancement_image, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()