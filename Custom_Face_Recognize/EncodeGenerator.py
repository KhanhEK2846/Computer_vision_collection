import cv2 as cv
import face_recognition as fr
import pickle as pk
import os

folderPath = 'images'
pathlist = os.listdir(folderPath)
imgList = []
studentIDs = []
for path in pathlist:
    imgList.append(cv.imread(os.path.join(folderPath,path)))
    studentIDs.append(os.path.splitext(path)[0])

def findEncoding(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

print("Started.....")
encodeListKnown = findEncoding(imgList)
encodeListKnownwithIDs = [encodeListKnown,studentIDs]

file = open("EncodeFile.p",'wb')
pk.dump(encodeListKnownwithIDs,file)
file.close()
print("Done")