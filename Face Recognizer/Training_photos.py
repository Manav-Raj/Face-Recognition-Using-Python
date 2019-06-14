import cv2,os
import numpy as np
from PIL import Image #PIL is python Image  Library all images processing works are done by it

path='Sample' # parh where sample images are stored

#Local Binary Pattern Histograms(LBPH)
recognizer = cv2.createLBPHFaceRecognizer();
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L');# convert the image into grayscale
        imageNp=np.array(pilImage,'uint8')# convert the pixel to unsigned integer
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        print Id
        Ids.append(Id)
        cv2.imshow("Trained_Samples",imageNp)
        cv2.waitKey(10)
    return Ids,faces


Ids,faces = getImagesAndLabels(path)
recognizer.train(faces, np.asarray(Ids))
recognizer.save('recognizer\\trainnerdata.yml')#store the traned image file to recognizer folder
cv2.destroyAllWindows()
