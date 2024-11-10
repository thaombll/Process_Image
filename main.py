import cv2
from Clustering_filter import Clustering_Filter 
import prepare_lowpass_downsampled_input as pre
from Interpolation_filter import EdgeSensitiveInterpolator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "lenna.png"
    original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    degraded_image = pre.lowpass_and_downsample(original_image, sigma=1.0, factor=2)

    #Interpolation_filter
    interpolator = EdgeSensitiveInterpolator(k = 1.0)
    proposed_image = interpolator.proposed_interpolation(degraded_image)


    height, width = proposed_image.shape
    reconstructed_image = Clustering_Filter(proposed_image, height, width, 0.5, 5, 40, 0.5)
    reconstructed_image.process_image()
    
