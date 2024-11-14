from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as psnr

class Prepare_data:
    def __init__(self, image, sigma):
        self.image = image
        self.sigma = sigma

    def lowpass_and_downsample(self):
        filtered_image = gaussian_filter(self.image, sigma=self.sigma)
        input_image = filtered_image[::2, ::2]
        return input_image