import cv2
import numpy as np
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt

class Clustering_Filter:
    def __init__(self, gray_image, height, width, alpha, k, size_neighborhood, s):
        self.gray_image = gray_image
        self.height = height
        self.width = width
        self.alpha = alpha
        self.k = k 
        self.size_neighborhood = size_neighborhood
        self.s = s
    
    def expand(self, a):
        image_expand = cv2.copyMakeBorder(self.gray_image, int(a), int(a), int(a), int(a), cv2.BORDER_REPLICATE)
        return image_expand

    def calculate_w(self):
        list_w = []
        sum_w = 0
        for i in range(3):
            for j in range(3):
                norm = np.linalg.norm(np.array([i, j]) - np.array([1, 1]))
                w = np.exp((-1) * self.alpha * norm)
                list_w.append(w)
                sum_w += w 
        return list_w, sum_w
    
    def calculate_y_(self, list_w, sum_w, window):
        multi, index = 0, 0
        for i in range(3):
            for j in range(3):
                multi += window[i][j] * list_w[index]
                index += 1
        return multi / sum_w 
    
    def calculate_sigma_2(self, y_, list_w, sum_w, window):
        multi, index = 0, 0
        for i in range(3):
            for j in range(3):
                multi += ((window[i][j] - y_) ** 2 ) * list_w[index]
                index += 1
        return multi / sum_w
    
    def calculate_beta(self, sigma_2):
        return 1 / (2 * (sigma_2 + 1e-10))
    
    def calculate_y(self, list_w, beta, y_, window):
        index = 0
        multi_x = 0
        for i in range(3):
            for j in range(3):
                multi_x += window[i][j] * list_w[index] * np.exp((-1) * beta * ((window[i][j] - y_) ** 2))
                index += 1 

        index = 0
        multi_y = 0
        for i in range(3):    
            for j in range(3):
                multi_y += list_w[index] * np.exp((-1) * beta * ((window[i][j] - y_) ** 2))
                index += 1 
        
        return multi_x / multi_y
    
    def calculate_output_y(self, list_w, beta, y_, window):
        output_y = y_
        cnt = self.k
        while cnt != 0:
            output_y = self.calculate_y(list_w, beta, output_y, window)
            cnt -=1
        return output_y

    def calculate_mean(self, neighborhood):
        return np.mean(neighborhood)
    
    def calculate_variance(self, neighborhood):
        return np.var(neighborhood)

    def enhancement_image(self, y, output_y, mean, variance):
        y_d = y - output_y
        y_m = 0
        if np.abs(y_d - mean) < 2.5 * variance:
            y_m = output_y
        else:
            y_m = y 
        
        y_o = y - self.s * y_m 
        
        return y_o

    def process_image(self):
        image_expand_1 = self.expand(1)
        image_expand_2 = self.expand(self.size_neighborhood / 2)

        list_window_1 =  view_as_windows(image_expand_1, (3, 3))
        list_window_2 =  view_as_windows(image_expand_2, (self.size_neighborhood, self.size_neighborhood))

        output_y = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                window = list_window_1[i][j]

                list_w, sum_w = self.calculate_w()
                y_ = self.calculate_y_(list_w, sum_w, window)
                sigma_2 = self.calculate_sigma_2(y_, list_w, sum_w, window)
                beta = self.calculate_beta(sigma_2)
                output_y[i][j] = self.calculate_output_y(list_w, beta, y_, window)

        mean = np.zeros((self.height, self.width))
        variance = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                neighborhood = list_window_2[i][j]
                mean[i][j] = self.calculate_mean(neighborhood)
                variance[i][j] = self.calculate_variance(neighborhood)

        enhancement_image = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                enhancement_image[i][j] = self.enhancement_image(self.gray_image[i][j], output_y[i][j], mean[i][j], variance[i][j])

        plt.imshow(enhancement_image, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()
        return enhancement_image