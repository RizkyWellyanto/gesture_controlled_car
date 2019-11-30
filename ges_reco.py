import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # For our purposes we don't need to detect hand
    # detect hand
    # hands_list = util.handDetect(candidate, subset, oriImg)

    # all_hand_peaks = []
    # for x, y, w, is_left in hands_list:
    #     peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     all_hand_peaks.append(peaks)

    # canvas = util.draw_handpose(canvas, all_hand_peaks)

    # A window to show the original video
    cv2.imshow('ges_reco', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()