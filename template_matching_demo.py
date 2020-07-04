import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging
import time

MIN_MATCH_COUNT = 10
FLANN_INDEX_LSH = 6
SCAN_RATIO = 0.5


def main():
    logger = logging.getLogger(__name__)

    prior_check = ()

    for i in range(10):
        img1 = cv2.imread('test/logo_l.jpg', 0)  # queryImage
        img2 = cv2.imread('test/logo_s.jpg', 0)  # trainImage
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # brisk_detector(img1, img2)

        img3, prior_check = detect_small_img(img1, img2, prior_check)
        if prior_check:
            cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('detector', 800, 600)
            cv2.imshow('detector', img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('output1.png', img3)


def detect_small_img(img1, img2, prior_check):
    start = time.time()
    img_r = img2.copy()
    if prior_check:
        ld_y = prior_check[1][1]
        ru_y = prior_check[0][1]
        ld_x = prior_check[1][0]
        ru_x = prior_check[0][0]

        img_l = img1[int(ld_y):int(ru_y), int(ld_x):int(ru_x)].copy()
        h1, w1 = img_l.shape
        h2, w2 = img_r.shape
        print('got prior_check: {}, {}, {}, {}'.format(ld_y, ru_y, ld_x, ru_x))
    else:
        img_l = img1.copy()
        h1, w1 = img_l.shape
        h2, w2 = img_r.shape
        ld_y = 0
        ru_y = h1
        ld_x = 0
        ru_x = w1

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img_l, None)
    # img_l - train img
    kp2, des2 = orb.detectAndCompute(img_r, None)
    # img_r - query img

    # Initialize and use FLANN
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)

    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    flag = 0
    avg_xl = 0
    avg_yl = 0
    avg_xr = 0
    avg_yr = 0
    count = 0
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
                if m.distance < SCAN_RATIO * n.distance:
                    count += 1
                    avg_xl += kp1[m.queryIdx].pt[0]
                    avg_yl += kp1[m.queryIdx].pt[1]
                    avg_xr += kp2[m.trainIdx].pt[0]
                    avg_yr += kp2[m.trainIdx].pt[1]
                    if flag == 0:
                        max = (m.queryIdx, m.trainIdx)
                        min = (m.queryIdx, m.trainIdx)
                        flag = 1
                    if kp1[m.queryIdx].pt[0] > kp1[max[0]].pt[0]:
                        max = (m.queryIdx, m.trainIdx)
                        # print(kp1[m.queryIdx].pt[0])
                    if kp1[m.queryIdx].pt[0] < kp1[min[0]].pt[0]:
                        min = (m.queryIdx, m.trainIdx)
                        # print(kp1[m.queryIdx].pt[0])
                        # pt1 = kp1[m.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
                        # pt2 = kp2[m.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号

    except:
        print("an error happened, resetting...")
        pass

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
        matchesMask = mask.ravel().tolist()

        # h,w = img1.shape
        pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img_l = cv2.polylines(img_l, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('detector', 800, 300)
        # cv2.imshow('detector', img_l)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if count == 0:
            count = 1

        avg_xl = avg_xl / count
        avg_yl = avg_yl / count
        avg_xr = avg_xr / count
        avg_yr = avg_yr / count
        pt1 = kp1[max[0]].pt
        pt2 = kp1[min[0]].pt
        pt3 = kp2[max[1]].pt
        pt4 = kp2[min[1]].pt

        # print(h1, w1, h2, w2)
        dis_rate = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) / np.sqrt(
            (pt3[0] - pt4[0]) ** 2 + (pt3[1] - pt4[1]) ** 2)
        if prior_check:
            print("prior check value")
            print(ld_x, ld_y, ru_x, ru_y)
            ru_x_new = dis_rate * (w2 - avg_xr) + avg_xl + ld_x
            ru_y_new = dis_rate * (h2 - avg_yr) + avg_yl + ld_y
            ld_x_new = avg_xl - dis_rate * avg_xr + ld_x
            ld_y_new = avg_yl - dis_rate * avg_yr + ld_y
        else:
            ru_x_new = dis_rate * (w2 - avg_xr) + avg_xl
            ru_y_new = dis_rate * (h2 - avg_yr) + avg_yl
            ld_x_new = avg_xl - dis_rate * avg_xr
            ld_y_new = avg_yl - dis_rate * avg_yr

        # print(dis_rate)
        print(tuple(dst[1][0]))
        # img_detect = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        img_detect = cv2.rectangle(img1, (int(ru_x_new), int(ru_y_new)), (int(ld_x_new), int(ld_y_new)), (255, 255, 0), 2).copy()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img_l, kp1, img_r, kp2, good, None, **draw_params)
        end = time.time()
        time_cost = (end - start) * 1000
        print("cost:%.4f ms" % time_cost)

        # cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('detector', 800, 300)
        # cv2.imshow('detector', img_detect)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        end = time.time()
        time_cost = (end - start) * 1000
        print("cost:%.4f ms" % time_cost)

    print("cost:%.4f ms" % time_cost)
    if len(good) > MIN_MATCH_COUNT:
        extended_value = 50
        # return img3, [(0, 0),(0, 0)]
        return img3, [(ru_x_new+extended_value, ru_y_new+extended_value),
                      (ld_x_new-extended_value, ld_y_new-extended_value)]
    else:
        return 0, ()


def brisk_detector(img1, img2):
    brisk = cv2.BRISK_create()
    kpt1, desc1 = brisk.detectAndCompute(img1, None)
    kpt2, desc2 = brisk.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # matches.sort(None, None, True)
    out_img = cv2.drawMatches(img1, kpt1, img2, kpt2, good, None)

    cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detector', 800, 600)
    cv2.imshow('detector', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('output1.png', out_img)


if __name__ == '__main__':
    main()



"""
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=0)
resultImage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawParams)

end = time.time()
time_cost = (end-start)*1000
print("cost:%.4f ms" %(time_cost))
cv2.namedWindow('Flann', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Flann', 400, 600)
cv2.imwrite('output.png', resultImage)
cv2.imshow('Flann', resultImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    plt.imshow(newimg)
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
"""


