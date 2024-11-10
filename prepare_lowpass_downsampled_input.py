import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as psnr


def lowpass_and_downsample(image, sigma=1.0, factor=2):
    # Áp dụng bộ lọc Gaussian để làm mờ (lọc thông thấp)
    filtered_image = gaussian_filter(image, sigma=sigma)
    # Giảm chiều ảnh (lấy mẫu thấp)
    #cv2.resize(filtered_image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    downsampled_image = filtered_image[::2, ::2]
    return downsampled_image