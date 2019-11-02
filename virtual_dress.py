
import numpy as np
import cv2
from PIL import Image

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height


while True:
    ret, img = cap.read()
    roi=np.zeros_like(img) #setting the shape for the image
    output=np.zeros_like(img)#setting the shape for the image
    d2=0
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,  minSize=(20, 20))
    d2 = Image.open('vest.png').convert('RGBA')
    
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.rectangle(img,(x-100,y+100),(x+w+80,y+h+260),(255,0,0),2)
        
        roi=img[y+80:y+h+260,x-80:x+w+80]
        output=img.copy()
        
        try:
            d2 = d2.resize((w+150,h+180), Image.ANTIALIAS)

            pilim = Image.fromarray(roi)
            pilim.paste(d2,box=(0,10),mask=d2)
            roi = np.array(pilim) # roi ma virtual dress banyo
    
        except:
        #    print('not found faces')
            pass
        #img[y:y+h,x:x+h]=0
        output[y+80:y+h+260,x-80:x+w+80]=roi
    #cv2.imshow('video',img)
    cv2.imshow('frame 2',output)
    
    #cv2.imshow('frame',roi)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'): # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
