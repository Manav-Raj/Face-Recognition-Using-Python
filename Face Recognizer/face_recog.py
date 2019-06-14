import cv2 #python library that deals with image processing using numpy
import numpy as np #numpy is a highly stable and fast array processing library of python.

# cv2 function which open the camera
cam = cv2.VideoCapture(0)
#This line Create the object of face and pass it to detector
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#This line Create the object of eye and pass it to eye_cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  

Id=raw_input('enter your id ')
count=0
while(True):
    #return the frame of image
    ret, img = cam.read()
    #it will convert image to Gray format because Gray format is easier for operation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, 1.3, 5) # 1.3 shows how much images size is reduced at each image scale and 5 is how much neigbours will be there in a window
    for (x,y,w,h) in faces:
        #this line will create rectangle frame around face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count=count+1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #this  line will write the image captured to sample folder in our project
        cv2.imwrite("Sample/User."+ str(Id) +'.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        #putText is use to show text on image here count is shown in our project
        cv2.putText(img,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        #this line will wait for 100 milisecond after a key is pressed and then close the window
        cv2.waitKey(100);
    cv2.imshow('My Face',img)
    if cv2.waitKey(100) & 0xFF == ord('q') or count==30:
        break
    else:
        print("Look at the camera")
#This statement will close the camera
cam.release()
#This line will close all windows
cv2.destroyAllWindows()
print("Sample Collection Completed Succesfully!!!")
