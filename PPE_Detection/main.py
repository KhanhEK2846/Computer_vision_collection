from ultralytics import YOLO
import cv2 as cv
import cvzone as cz
import math


cap = cv.VideoCapture("video/example2.mp4")
model = YOLO("YoLo_Weight/ppe.pt")
className =   [
    "Excavator",
    "Gloves",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "SUV",
    "Safety Cone",
    "Safety Vest",
    "bus",
    "dump truck",
    "fire hydrant",
    "machinery",
    "mini-van",
    "sedan",
    "semi",
    "trailer",
    "truck and trailer",
    "truck",
    "van",
    "vehicle",
    "wheel loader",
  ]

while True:
    success, img = cap.read()
    result = model (img,stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            #Bouding
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)  
            w,h = x2-x1,y2-y1
            cz.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0])*100)/100
            
            #class
            cls = int(box.cls[0])
            
            cz.putTextRect(img,f'{className[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1, thickness=1)
    cv.imshow("Image",img)
    if cv.waitKey(1) == 32:
        break
    
cap.release()
cv.destroyAllWindows()