import cv2
import numpy as np
from matplotlib import pyplot as plt

#特徴量比較関数
def match_features(img1, img2, detector='SIFT'):
    if detector == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif detector == 'ORB':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError("対応していない特徴量です")
    
    kp1, des1 = feature_detector.detectAndCompute(img1, None)
    kp2, des2 = feature_detector.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2 if detector == 'SIFT' else cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    return result, len(matches)

img1 = cv2.imread('test_park1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('test_park2.jpg', cv2.IMREAD_GRAYSCALE)

#SIFTで特徴量比較
result, num_matches = match_features(img1, img2, detector='SIFT')
plt.imshow(result), plt.title(f"SIFT Match: {num_matches} pts"), plt.axis('off')
plt.show()

