# Canny Edge Detector 구현
# 1. Noise Reduction
# 2. Compute x and y derivatives of image
# 3. Compute magnitude of gradient at evey pixel
# 4. Nommax_suppression
# 5. Hysteresis Thresholding(Triple)

import numpy as np
import cv2
import os

# Caany Edge Detector 클래스 선언 및 정의

class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, mid_pixel = 160, strong_pixel=255, lowthreshold=0.05, midthreshold=0.1, highthreshold=0.15):
        self.imgs = imgs # Canny Edge Detection 적용 전 이미지들
        self.imgs_final = [] # Canny Edge Detection 적용 후 이미지들
        self.img_smoothed = None # Smoothing 적용한 이미지
        self.gradientMat = None # gradient 크기 Matrix
        self.thetaMat = None # gradient 방향 Matrix
        self.nonMaxImg = None # non_max suppresion 적용한 이미지
        self.thresholdImg = None # Hysteresis Thresholding 적용한 이미지
        self.weak_pixel = weak_pixel # weak edge로 고려할 값
        self.mid_pixel = mid_pixel # mid edge로 고려할 값
        self.strong_pixel = strong_pixel # strong edge로 고려할 값
        self.sigma = sigma # Gaussian Filtering의 sigma 값
        self.kernel_size = kernel_size # 가우시안 블러링 커널 사이즈
        self.lowThreshold = lowthreshold # low threshold 값
        self.midThreshold = midthreshold # mid threshold 값
        self.highThreshold = highthreshold # high threshold 값 
        
        return 
    

    # 1. Noise Reduction
    # 2D Gaussian Smoothing Filter
    # (size x size) kernel with (sigma) 만들기

    def gaussian_kernel(self, size, sigma=1):
        # 2D (size x size) kernel 배열 초기화
        kernel = np.zeros((size, size))

        # 커널 중심 계산
        center = int(size) // 2
        
        # 2D Gaussian filter 수식
        for i in range(size):
            for j in range(size):
                x, y = (i - center), (j - center)
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma**2)
        
        # kernel 정규화
        kernel /= np.sum(kernel)
        
        return (kernel)
    

    # 2. Compute x and y derivatives of image
    # 3. Compute magnitude of gradient at every pixel
    # Sobel filter를 이용하여 x, y 방향으로 미분값 구하기
    # 미분값을 통해 gradient의 크기와 방향을 구함

    def sobel_filters(self, img):
        # sobel_filter 값
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # Convlution 수행
        Ix = convolution(img, Gx)
        Iy = convolution(img, Gy)

        # Gradient 크기 구하기
        # G_m = Gradient Magnitude
        G_m = np.hypot(Ix, Iy) # Ix, Iy 요소별로 유클리드 거리 계산
        G_m = G_m / G_m.max() * 255 # 정규화
        
        # Gradient 방향 구하기
        theta = np.arctan2(Iy, Ix)
        
        return (G_m, theta)
    

    # 4. Nommax_suppression
    # Eliminate of the magnitude in the direction of the gradient
    # gradient 방향으로 가장 maximum인 것 하나만 남기기(thinning)

    def non_max_suppression(self, gradient_m, gradient_o):
        # 이미지의 크기 구하기
        cols, rows = gradient_m.shape # gradient_m = gradient_magnitude
        result = np.zeros((cols, rows), dtype=np.int32) # 결과를 저장할 배열 정의
        angle = gradient_o * 180. / np.pi # gradient_o = gradient_orientation
        angle[angle < 0] += 180 # 방향을 0 ~ 180의 범위로 설정

        # 입력 이미지의 모든 픽셀을 반복
        # 이미지의 경계 픽셀은 제외
        for i in range(1, cols - 1):
            for j in range(1, rows - 1):
                orientation = angle[i, j]

                # 0도
                if (0 <= orientation < 22.5) or (157.5 <= orientation <= 180):
                    prev_p = gradient_m[i, j + 1] # prev_p = prev_pixel
                    next_p = gradient_m[i, j - 1] # next_p = next_pixel
                # 45도
                elif (22.5 <= orientation < 67.5):
                    prev_p = gradient_m[i + 1, j - 1]
                    next_p = gradient_m[i - 1, j + 1]
                # 90도
                elif (67.5 <= orientation < 112.5):
                    prev_p = gradient_m[i + 1, j]
                    next_p = gradient_m[i - 1, j]
                # 135도
                elif (112.5 <= orientation < 157.5):
                    prev_p = gradient_m[i - 1, j - 1]
                    next_p = gradient_m[i + 1, j + 1]

                # 현재 픽셀과 인접한 두 픽셀을 비교
                # 현재 픽셀의 gradient magnitude가 인접한 두 픽셀보다 큰 경우 result 배열에 해당 값을 유지
                # 작은 경우 에는 0으로 설정
                if (gradient_m[i, j] > prev_p) and (gradient_m[i, j] > next_p):
                    result[i, j] = gradient_m[i, j]

        return result


    # 5. Hysteresis Thresholding
    # Select the pixels such that M > T_h (high threshold)
    # Collect the pixels such that M > T_l (low threshold) that are neighnors of already collected edge points

    def threshold(self, img):
        # Triple Threshold 구하기
        highThreshold = img.max() * self.highThreshold
        midThreshold = img.max() * self.midThreshold
        lowThreshold = img.max() * self.lowThreshold

        # 결과 저장할 배열 정의
        result = np.zeros(img.shape, dtype=np.int32)

        # weak, mid, edge로 고려할 값 갖고오기
        weak = np.int32(self.weak_pixel)
        mid = np.int32(self.mid_pixel)
        strong = np.int32(self.strong_pixel)

        # strong edge 찾기
        # gradient_magnitude > highThreshold
        strong_i, strong_j = np.where(img > highThreshold)

        # mid edge 찾기
        # midThreshold < gradient_magnitude <= highThreshold
        mid_i, mid_j = np.where((img <= highThreshold) & (img > midThreshold))

        # weak edge 찾기
        # lowThreshold < gradient_magnitude <= midThreshold
        weak_i, weak_j = np.where((img <= midThreshold) & (img > lowThreshold))

        # non edge 찾기
        # gradient_magnitude < lowThreshold
        # 알아서 0 값으로 설정됨

        # edge 값 설정
        result[strong_i, strong_j] = strong
        result[mid_i, mid_j] = mid
        result[weak_i, weak_j] = weak

        return (result)
    

    # mid edge 주변의 weak edge 찾기
    # 위의 조건을 만족하는 weak edge는 mid edge라고 간주

    def hysteresis1(self, img):
        # 이미지 크기 갖고오기
        height, width = img.shape

        # weak, mid edge로 고려할 값 갖고오기
        weak = self.weak_pixel
        mid = self.mid_pixel

        # 입력 이미지의 모든 픽셀을 반복
        # 이미지의 경계 픽셀은 제외
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if (img[i, j] == weak):
                    # mid edge 주변의 weak edge 찾기
                    if ((img[i + 1, j - 1] == mid) or (img[i + 1, j] == mid)
                        or (img[i + 1, j + 1] == mid) or (img[i, j - 1] == mid)
                        or (img[i, j + 1] == mid) or (img[i - 1, j - 1] == mid)
                        or (img[i - 1, j] == mid) or (img[i - 1, j + 1] == mid)):
                        img[i, j] = mid
                    else:
                        img[i, j] = 0

        return (img)


    # strong edge 주변의 mid edge 찾기
    # 위의 조건을 만족하는 mid edge는 strong edge라고 간주

    def hysteresis2(self, img):
        # 이미지 크기 갖고오기
        height, width = img.shape
        
        # mid, strong edge로 고려할 값 갖고오기
        mid = self.mid_pixel
        strong = self.strong_pixel

        # 입력 이미지의 모든 픽셀을 반복
        # 이미지의 경계 픽셀은 제외
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if (img[i, j] == mid):
                    # strong edge 주변의 mid edge 찾기
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong) or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0

        return (img)


    # Canny Edge Detecion 수행

    def detect(self):
        img_final = []
        for i, img in enumerate(self.imgs):    
            self.img_smoothed = convolution(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            #print("smoothing" + str(i))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            #print("sobelfilter" + str(i))
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            #print("nonmax" + str(i))
            self.thresholdImg = self.threshold(self.nonMaxImg)
            #print("threshold" + str(i))
            img_hysteresis = self.hysteresis1(self.thresholdImg)
            img_final = self.hysteresis2(img_hysteresis)
            #print("hysteresis" + str(i))
            self.imgs_final.append(img_final)

        return (self.imgs_final)


# image와 kernel을 convolution하는 함수

def convolution(image, kernel):
    # 이미지에 패딩을 추가 (커널 크기에 따라 패딩값 설정)
    size = int((len(kernel) - 1) / 2)
    padding = np.pad(image, ((size, size), (size, size)), 'constant', constant_values=0)

    # kernel을 뒤집기
    kernel_r1 = np.rot90(kernel)
    kernel_r2 = np.rot90(kernel_r1)

    # 연산 결과를 담을 배열 초기화
    img_h, img_w = image.shape
    output = np.ones((img_h, img_w))
    output = output.astype('float32')
    
    # Convolution 연산 수행
    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = np.sum(padding[i:i + len(kernel), j:j + len(kernel)] * kernel_r2)
    
    return (output)


# Canny Edge Detection process 수행

def process_imgs(input_folder, output_folder):
    # 입력 폴더의 이미지 파일 목록 가져오기
    imgs_files = os.listdir(input_folder)

    # Canny Edge Detector 클래스 인스턴스 선언
    detector = cannyEdgeDetector([])

    for img_file in imgs_files:
        file_name, file_extension = os.path.splitext(img_file)
        if file_extension.lower() == '.jpg': # 파일 확장자가 JPG인 것만 고려함
            img_path = os.path.join(input_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 이미지를 흑백으로 읽어옴
            detector.imgs.append(img) # Canny Edge Detector 클래스의 imgs 리스트에 이미지 추가
    
    # Canny Edge Detection 수행
    edges = detector.detect()

    # 결과 이미지를 다른 폴더에 저장
    for idx, edge_img in enumerate(edges):
        output_path = os.path.join(output_folder, f"canny_edge_{idx}.png")
        cv2.imwrite(output_path, edge_img) # Caany Edge Detection 수행한 후의 이미지 저장


# main 함수

if __name__ == '__main__':
    input_imgs_path = 'test/' # 입력 이미지가 있는 폴더 경로
    output_imgs_path = 'test_output/' # 출력 이미지를 저장할 폴더 경로

    # Canny Edge Detection 수행
    process_imgs(input_imgs_path, output_imgs_path)