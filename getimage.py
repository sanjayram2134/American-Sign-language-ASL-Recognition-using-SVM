import numpy as np
import cv2 as cv
from pathlib import Path


def get_image():
    Letter = 'Y'
    Path('DATASET/' + Letter).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0
    while True:

        ret, frame = cap.read()

        if not ret:
            print("cannot capture")
            break
        i += 1

        cv.imwrite('DATASET/' + Letter + '/' + str(i) + '.png', frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i > 150:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    get_image()
