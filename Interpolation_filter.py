import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as psnr
import Prepare_data as pre

class EdgeSensitiveInterpolator:
    def __init__(self, k=1.0):
        self.k = k

    def _nonlinear_interpolation_1d(self, a, b, c, d):
        # Chuyển đổi các giá trị sang kiểu float để tránh overflow
        a, b, c, d = float(a), float(b), float(c), float(d)
        mu = (self.k * ((c - d) ** 2 + 1)) / (self.k * ((a - b) ** 2 + (c - d) ** 2) + 2)
        x = mu * b + (1 - mu) * c
        return x

    def proposed_interpolation(self, image):
        image = image.astype(np.float64)
        h, w = image.shape
        interpolated_image = np.zeros((h * 2, w * 2), dtype=float)

        # Nội suy theo hướng ngang
        for y in range(h):
            for x in range(w):
                b = image[y, x]
                c = image[y, min(x + 1, w - 1)]
                if x > 0 and x < w - 2:
                    a = image[y, x - 1]
                    d = image[y, min(x + 2, w - 1)]
                    interpolated_image[2 * y, 2 * x + 1] = self._nonlinear_interpolation_1d(a, b, c, d)
                else:
                    interpolated_image[2 * y, 2 * x + 1] = (b + c) / 2
                interpolated_image[2 * y, 2 * x] = b

        # Nội suy theo hướng dọc
        for y in range(h):
            for x in range(w):
                b = image[y, x]
                c = image[min(y + 1, h - 1), x]
                if y > 0 and y < h - 2:
                    a = image[y - 1, x]
                    d = image[min(y + 2, h - 1), x]
                    interpolated_image[2 * y + 1, 2 * x] = self._nonlinear_interpolation_1d(a, b, c, d)
                else:
                    interpolated_image[2 * y + 1, 2 * x] = (b + c) / 2

        # Tính giá trị trung bình cho các điểm z
        for y in range(h):
            for x in range(w):
                cnt = 2
                interpolated_image[2 * y + 1, 2 * x + 1] += interpolated_image[2 * y, 2 * x + 1] + interpolated_image[2 * y + 1, 2 * x]
                if 2 * x + 2 < 2 * w:
                    interpolated_image[2 * y + 1, 2 * x + 1] += interpolated_image[2 * y + 1, 2 * x + 2]
                    cnt += 1
                if 2 * y + 2 < 2 * h:
                    interpolated_image[2 * y + 1, 2 * x + 1] += interpolated_image[2 * y + 2, 2 * x + 1]
                    cnt += 1
                interpolated_image[2 * y + 1, 2 * x + 1] /= cnt

        return np.clip(interpolated_image, 0, 255).astype(np.uint8)

    # Nội suy tuyến tính
    def linear_interpolation(self, image):
        return cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

    # Nội suy bậc ba
    def cubic_interpolation(self, image):
        return cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_CUBIC)


def calculate_psnr_for_detail_areas(original_image, interpolated_image, theta):
    detail_mask = cv2.Laplacian(original_image, cv2.CV_64F) ** 2 > theta
    detail_pixels = np.where(detail_mask)
    detail_pixels = (np.clip(detail_pixels[0], 0, interpolated_image.shape[0] - 1),
                     np.clip(detail_pixels[1], 0, interpolated_image.shape[1] - 1))
    original_detail_pixels = original_image[detail_pixels]
    interpolated_detail_pixels = interpolated_image[detail_pixels]
    return psnr(original_detail_pixels, interpolated_detail_pixels)


def evaluate_interpolators(original_image, theta_values, linear_interpolated, cubic_interpolated, proposed_interpolated):
    psnr_linear = []
    psnr_cubic = []
    psnr_proposed = []
    for theta in theta_values:
        psnr_linear.append(calculate_psnr_for_detail_areas(original_image, linear_interpolated, theta))
        psnr_cubic.append(calculate_psnr_for_detail_areas(original_image, cubic_interpolated, theta))
        psnr_proposed.append(calculate_psnr_for_detail_areas(original_image, proposed_interpolated, theta))
    return psnr_linear, psnr_cubic, psnr_proposed


def plot_psnr(theta_values, psnr_linear, psnr_cubic, psnr_proposed):
    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, psnr_proposed, label="Proposed Interpolator", linestyle="-")
    plt.plot(theta_values, psnr_linear, label="Linear", linestyle="--")
    plt.plot(theta_values, psnr_cubic, label="Cubic", linestyle=":")
    plt.xlabel("Variance Threshold (θ)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR (evaluated in detail areas) vs. Variance Threshold θ")
    plt.xlim(left=0)
    plt.ylim(bottom=min(min(psnr_linear), min(psnr_cubic), min(psnr_proposed)) - 1)
    plt.legend()
    plt.grid(False)
    plt.show()


def show_interpolated_images(original_image, linear_image, cubic_image, proposed_image):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.title("Linear Interpolation")
    plt.imshow(linear_image, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title("Cubic Interpolation")
    plt.imshow(cubic_image, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title("Proposed Interpolation")
    plt.imshow(proposed_image, cmap='gray')
    plt.tight_layout()
    plt.show()


# def test_qualified_methods():
#     path = "lenna.png"
#     original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Sử dụng cv2.imread
#     degraded_image = pre.lowpass_and_downsample(original_image, sigma=1.0, factor=2)

#     interpolator = EdgeSensitiveInterpolator(k=1.0)
#     linear_image = interpolator.linear_interpolation(degraded_image)
#     cubic_image = interpolator.cubic_interpolation(degraded_image)
#     proposed_image = interpolator.proposed_interpolation(degraded_image)

#     show_interpolated_images(original_image, linear_image, cubic_image, proposed_image)

#     theta_values = np.arange(0, 800, 50)
#     psnr_linear, psnr_cubic, psnr_proposed = evaluate_interpolators(original_image, theta_values, linear_image, cubic_image, proposed_image)

#     plot_psnr(theta_values, psnr_linear, psnr_cubic, psnr_proposed)


# test_qualified_methods()