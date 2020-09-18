import cv2
import numpy as np
import math

#read from primary camera
cap = cv2.VideoCapture(0)
width = cap.get(3)  # float
height = cap.get(4)  # float
print(f"width = {width} height = {height}")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

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
        roi = frame[0:200, 200:480]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        cv2.rectangle(frame, (0, 200), (640,480), (0, 255, 0), 3)
        cv2.line(frame, (0, 0), (640, 480), (255, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        # define range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # find contours
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except Exception as e:
        print(e)
        wait = input("sgdgsd")
        print("Eoor")
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
