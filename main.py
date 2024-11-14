import cv2
from Clustering_filter import Clustering_Filter 
from Prepare_data import Prepare_data
from Interpolation_filter import EdgeSensitiveInterpolator
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    list_path = ["image/aa.webp", "image/cameraman.png", "image/lenna.png", "image/1.jpg", "image/example.jpg", "image/license.jpg"]

    for i in range(len(list_path)):
        path = r"{}".format(list_path[i])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        pre_data = Prepare_data(image, 1.0)
        input_image = pre_data.lowpass_and_downsample()
        original_image_path = os.path.normpath(os.path.join(r"Ouput/Original_Image", f"image{i}.png"))
        plt.imshow(input_image, cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig(original_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        interpolator = EdgeSensitiveInterpolator(k = 1.0)
        input_clustering_filter = interpolator.proposed_interpolation(input_image)
        interpolator_filter_path = os.path.normpath(os.path.join(r"Ouput/Interpolation_filter", f"image{i}.png"))
        plt.imshow(input_clustering_filter, cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig(interpolator_filter_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        height, width = input_clustering_filter.shape
        reconstructed_image = Clustering_Filter(input_clustering_filter, height, width, 0.5, 5, 40, 0.5)
        output_image = reconstructed_image.process_image()
        result_image_path = os.path.normpath(os.path.join(r"Ouput/Result_Image", f"image{i}.png"))
        plt.imshow(output_image, cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
