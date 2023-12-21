import cv2
import numpy as np

# 두 개의 Canny 엣지 이미지 로드
canny_image_1 = cv2.imread('test_output/canny_edge_0.png', cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 로드
canny_image_2 = cv2.imread('test_output/canny_edge_1.png', cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 로드

# 두 이미지 간의 차이 계산
difference = cv2.absdiff(canny_image_1, canny_image_2)
cv2.imwrite('output/differ.png', difference)