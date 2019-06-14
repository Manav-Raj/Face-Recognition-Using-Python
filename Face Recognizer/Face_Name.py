import cv2
import numpy as np
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainnerdata.yml") # Load the trained photos stored
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4) #This will print the name
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convert the image to gray scale
    faces = detector.detectMultiScale(gray, 1.3, 6)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])# Prdeict the image by camera
        if(id==45):
            id="Vishal Bhaiya"
        if(id==2):
            id="Partha Sir"
        if(id==3):
            id="Chayan"
        if(id==114):
            id="Riya"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
  
cam.release()
cv2.destroyAllWindows()
 
