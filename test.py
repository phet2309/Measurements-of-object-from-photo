from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from cv2.cv2 import drawContours


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


pixelsPerMetric = None
# img=cv2.imread('G:\Sem-8\Project\object sorting and packaging\extra\image1.jpg',-1)
img = cv2.imread('D:\project\photo_pixel\photo.jpg', -1)
# rsz = cv2.imread('D:\project\photo_pixel\photo.jpg', -1)
# img1=cv2.imread('G:\Sem-8\Project\object sorting and packaging\extra\image1.jpg',-1)
dsz = (600, 500)
# dsz = (3000, 4000)
# dsz1=(600,500)
rsz = cv2.resize(img, dsz)
# rsz1= cv2.resize(img1,dsz1)

hsv = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)
# hsv1 = cv2.cvtColor(rsz1,cv2.COLOR_BGR2HSV)

# referance maak
lower_yellow = np.array([5, 100, 50])
higher_yellow = np.array([30, 234, 255])

yellow_mask = cv2.inRange(hsv, lower_yellow, higher_yellow)
# yellow_mask1 = cv2.inRange(hsv1,lower_yellow,higher_yellow)

# diameter mask
lower_red = np.array([0, 60, 70])
higher_red = np.array([10, 200, 250])

red_mask = cv2.inRange(hsv, lower_red, higher_red)
cv2.imshow('red-mask', red_mask)
cv2.imshow('yellow-mask', yellow_mask)
cnts1, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts2, hierarchy = cv2.findContours(yellow_mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntss1, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(cntss1))
print(type(cntss1))

for pic, contour1 in enumerate(cnts1):
    area_yellow = cv2.contourArea(contour1)
    if area_yellow > 5000:
        rect = cv2.minAreaRect(contour1)
        box = cv2.boxPoints(rect)
        box = np.array(box,dtype=int)
        box = order_points(box)
        cv2.drawContours(rsz, [contour1], 0, (0, 242, 255), 2)

        for (x,y) in box:
            cv2.circle(rsz, (int(x), int(y)), 5, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(rsz, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(rsz, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            cv2.line(rsz, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric=dB/300
                # pixelsPerMetric = dB / args["width"]

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            cv2.putText(rsz, "{:.1f}mm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
            cv2.putText(rsz, "{:.1f}mm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)



for pic, contour1 in enumerate(cnts1):
    area_yellow = cv2.contourArea(contour1)
    if area_yellow > 5000:
        rect = cv2.minAreaRect(contour1)
        box = cv2.boxPoints(rect)
        box = np.array(box,dtype=int)
        box = order_points(box)
        cv2.drawContours(rsz, [contour1], 0, (0, 242, 255), 2)

        for (x,y) in box:
            cv2.circle(rsz, (int(x), int(y)), 5, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(rsz, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(rsz, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            cv2.line(rsz, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric=dB/300
                # pixelsPerMetric = dB / args["width"]

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            cv2.putText(rsz, "{:.1f}mm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
            cv2.putText(rsz, "{:.1f}mm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)



for pic, contour in enumerate(cntss1):
    area_red = cv2.contourArea(contour)
    if area_red > 5000:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box,dtype=int)
        box = order_points(box)
        cv2.drawContours(rsz, [contour], 0, (0, 242, 255), 2)

        for (x,y) in box:
            cv2.circle(rsz, (int(x), int(y)), 5, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(rsz, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(rsz, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(rsz, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            cv2.line(rsz, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric=dB/300
                # pixelsPerMetric = dB / args["width"]

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            cv2.putText(rsz, "{:.1f}mm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
            cv2.putText(rsz, "{:.1f}mm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)






cv2.imshow('final',rsz)
cv2.waitKey(0)
