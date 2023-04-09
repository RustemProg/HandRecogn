import cv2
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

def outHand(fingerup, img):
# numbers
    if fingerup == [0, 0, 0, 0, 0]:
        cv2.putText(img, str(0), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (50,255,50), 12)
    elif fingerup == [0, 1, 0, 0, 0]:
        cv2.putText(img, str(1), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [0, 1, 1, 0, 0]:
        cv2.putText(img, str(2), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [1, 1, 1, 0, 0]:
        cv2.putText(img, str(3), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [1, 1, 1, 1, 1]:
        cv2.putText(img, str(5), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [0, 1, 1, 0, 1]:
        cv2.putText(img, str(7), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
#letters
    elif fingerup == [1, 0, 0, 0, 0]:
        cv2.putText(img, str('A'), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [0, 0, 0, 0, 1]:
        cv2.putText(img, str('I'), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [1, 1, 0, 0, 0]:
        cv2.putText(img, str('L'), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    elif fingerup == [1, 0, 0, 0, 1]:
        cv2.putText(img, str('Y'), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)

    if fingerup == [0, 1, 1, 1, 1]:
        cv2.putText(img, str(4), (100,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    if fingerup == [0, 1, 1, 1, 0]:
        cv2.putText(img, str(6), (120,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    if fingerup == [0, 1, 0, 1, 1]:
        cv2.putText(img, str(8), (100,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    if fingerup == [0, 0, 1, 1, 1]:
        cv2.putText(img, str(9), (110,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)

    if fingerup == [0, 1, 1, 1, 1]:
        cv2.putText(img, str('B'), (210,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    if fingerup == [0, 0, 1, 1, 1]:
        cv2.putText(img, str('F'), (210,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
    if fingerup == [0, 1, 1, 1, 0]:
        cv2.putText(img, str('W'), (210,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)

cap = cv2.VideoCapture(0)
cap.set(3, 1980)
cap.set(4, 1080)
cap.set(10, 100)


mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
npDraw = mp.solutions.drawing_utils
detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    multiLandMarks = results.multi_hand_landmarks
    hand = detector.findHands(img, draw=False)
    if hand:
        lmlist = hand[0]
        if lmlist:
            fingerup = detector.fingersUp(lmlist) 
            outHand(fingerup, img)
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f'HAND NUMBER: {hand_no+1}')
            print('-----------------------')
            for i in range(21):
                print(f'{mpHands.HandLandmark(i).name}:')
                print(f'{hand_landmarks.landmark[mpHands.HandLandmark(i).value]}')
            print('End of a numbers')
        for handLms in results.multi_hand_landmarks:
            npDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow('python', img)
    if cv2.waitKey(20) == 27 and cv2.waitKey(0):
        break

cv2.destroyWindow("python")
cap.release()
cv2.waitKey(1)