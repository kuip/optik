import numpy as np
import cv2
import time


cap = cv2.VideoCapture(0)

def find_circles(sigma=0.22):
    while(True):
        # ret -> boolean
        # img -> image matrix
        ret, img = cap.read()
        cimg = img.copy()

        gray = cv2.cvtColor(img,
            cv2.COLOR_RGB2GRAY,
        )

        gray = cv2.medianBlur(gray, 5)
        kernel = np.ones((3,3),np.uint8)

        gray = cv2.dilate(gray, kernel, iterations=2)
        cv2.imshow('after dilation', gray)

        gray = cv2.erode(gray, kernel, iterations=2)
        cv2.imshow('after erosion', gray)

        # compute the median of the single channel pixel intensities
        v = np.median(gray)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        gray = cv2.Canny(gray, lower, upper)

        # TODO: ajust Canny
        cv2.imshow('after canny', gray)

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
            for i in circles[0, :]:
                cv2.circle(cimg, (i[0],i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255), 3)

            cv2.imshow('frame', cimg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    find_circles()
