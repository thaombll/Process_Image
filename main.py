import cv2
from Clustering_filter import Clustering_Filter 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = plt.imread("aa.webp")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    pre_image = Clustering_Filter(gray_image, height, width, 0.5, 5, 40, 0.5)
    pre_image.process_image()
    
