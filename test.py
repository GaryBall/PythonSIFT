import cv2
from surf_ori import FeatureMatching

img_train = cv2.imread('test/2.jpg')

matching = FeatureMatching(query_image='test/1.jpg')
flag = matching.match(img_train)
print(flag)
