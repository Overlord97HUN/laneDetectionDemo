import cv2
import imagepreprocessor as pre
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture('sample_videos/lane_detection_test_video.mp4')

    counter = 0
    frame_selected = 5

    while cap.isOpened():
        ret, frame = cap.read()

        processed_frame = pre.processedImage(frame)

        if ret:
            cv2.imshow('frame', processed_frame)
            cv2.imshow('original', frame)
            if counter == frame_selected:
                cv2.imwrite('sample_videos/test_image_%s.jpg' % str(frame_selected), frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()
