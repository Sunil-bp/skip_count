import cv2
import numpy as np
import math
import traceback
import sys


#read from primary camera
cap = cv2.VideoCapture(0)
width = cap.get(3)  # float
height = cap.get(4)  # float
print(f"width = {width} height = {height}")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
y2 = []
x2 = []
buffer = list(np.ones(20))
intwe = 0
peek = 0
pre_peek = 0
count = 0
while (1):
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)
        # define region of interest
        roi = frame[200:480, 0:640]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        cv2.rectangle(frame, (0, 200), (640,480), (0, 255, 0), 3)
        cv2.line(frame, (0, 0), (640, 480), (255, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(count), (450, 80), font, 3, (255, 255, 255), 1, cv2.LINE_AA)
        # define range of skin color in HSV
        lower_skin = np.array([0, 44, 71], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # find contours
        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour of max area(hand)
        try:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
        except:
            print("no legs")
            continue

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        color_image = np.zeros((312, 512, 3), np.uint8)

        #draw around my legs
        cv2.drawContours(color_image, [cnt], 0, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(cnt)
        print(y+h)

        #find peak
        buffer.append(y+h)
        buffer.pop(0)
        if max(buffer) in [buffer[4],buffer[5],buffer[6]]:
            peek = max(buffer)
        if peek != pre_peek:
            count += 1
        pre_peek = peek

        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(color_image, (0, y+h), (640, y+h), (255, 0, 0), 1)

        areacnt = cv2.contourArea(cnt)
        cv2.imshow('mask', mask)
        cv2.imshow('legs', color_image)
        cv2.imshow('frame', frame)
    except Exception as e:
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])
        wait = input("sgdgsd")
        print("Eoor")
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
