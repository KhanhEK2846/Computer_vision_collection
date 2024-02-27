import cv2 as cv
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
detector = HandDetector()

while True:
    success,img = cap.read()
    
    if success:
        hands,img_out = detector.findHands(img)
        cv.imshow('img',img_out)
    
    if cv.waitKey(1) == 32:
        break
    
    
    
cap.release()
cv.destroyAllWindows()