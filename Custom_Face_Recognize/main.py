import cv2 as cv
import pickle as pk
import face_recognition as fr
import numpy as np
import cvzone as cz

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#Load the encoding file
file = open('EncodeFile.p','rb')
encodeListKnownwithIDs = pk.load(file)
file.close()
encodeListKnown,studentIDs = encodeListKnownwithIDs


while True:
    success,img = cap.read()
    
    imgS = cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS,cv.COLOR_BGR2RGB)
    
    faceCurFrame = fr.face_locations(imgS)
    encodeCurFrame = fr.face_encodings(imgS,faceCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = fr.compare_faces(encodeListKnown,encodeFace)
        faceDis = fr.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            bbox = x1,y1,x2-x1,y2-y1
            img = cz.cornerRect(img,bbox,rt=0)
        
    cv.imshow("Face",img)
    if cv.waitKey(1) == 32:
        break
    
    
    
cap.release()
cv.destroyAllWindows()