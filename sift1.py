import cv2 
import matplotlib.pyplot as plt

img1 = cv2.imread('a2.jpg')  
img2 = cv2.imread('a1.jpg') 

img1 = cv2.imread('brain2/train/yes/Y1.jpg')  
img2 = cv2.imread('brain2/train/yes/Y3.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1), plt.xticks([]), plt.yticks([]), plt.imshow(cv2.drawKeypoints(img1, keypoints_1, img1.copy())) 
plt.subplot(1,2,2), plt.xticks([]), plt.yticks([]), plt.imshow(cv2.drawKeypoints(img2, keypoints_2, img2.copy()))
plt.figure(figsize=(10,10))
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:15], img2, flags=2)
plt.imshow(img3) ,plt.xticks([]), plt.yticks([]), plt.title('SIFT matching'), plt.imshow(img3)

