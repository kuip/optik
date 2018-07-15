import numpy as np
import cv2
import time


cap = cv2.VideoCapture(0)

def opening(image, interations):
    kernel = np.ones((3,3),np.uint8)
    image = cv2.dilate(image,
        kernel,
        interations,
    )
    image = cv2.erode(image,
        kernel,
        interations,
    )
    return image


def canny_edge_detection(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        image = cv2.Canny(image,
            lower,
            upper,
        )
        return image

def find_circles():
    while(True):
        # ret -> boolean
        # img -> image matrix
        ret, img = cap.read()
        cimg = img.copy()

        gray = cv2.cvtColor(img,
            cv2.COLOR_RGB2GRAY,
        )

        gray = cv2.medianBlur(gray, 5)
        gray = opening(gray, interations=1)
        cv2.imshow('After opening', gray)

        gray = canny_edge_detection(gray)
        cv2.imshow('After canny', gray)

        circles = cv2.HoughCircles(gray,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
        )
        try:
            circles = np.uint16(np.around(circles))
        except :
            print('no circles')
        else:
            # when two circles exactly are detected, draw a line between them
            if len(circles[0]) == 2:
                cv2.line(cimg, (circles[0][0][0], circles[0][0][1]),(circles[0][1][0], circles[0][1][1]), (255, 0, 0), 3)
            for i in circles[0, :]:
                cv2.circle(cimg, (i[0],i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(cimg, (i[0],i[1]), 1, (0,0,255), 3)

            cv2.imshow('frame', cimg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    find_circles()
