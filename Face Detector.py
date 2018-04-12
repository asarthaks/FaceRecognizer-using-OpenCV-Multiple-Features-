import cv2
import matplotlib.pyplot as plt

#defining the classes

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only = True) :
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30,30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE

        faces_coord = self.classifier.detectMultiScale(frame, scaleFactor= scale_factor, minNeighbors = min_neighbors,
                                        minSize = min_size, flags = flags)

        return faces_coord


#initializing the camera
webcam = cv2.VideoCapture(0)
print(webcam.isOpened())

#initializing the detector
detector = FaceDetector("haarcascade_frontalface_default.xml")
detector2 = FaceDetector("frontalEyes35x16.xml")



#drawing rectangle and displaying frame
while webcam.isOpened():
        ret,frame = video.read()    
        frame = cv2.flip(frame,1)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE

        faces_coord = detector.detectMultiScale(gray,1.2,5,minSize=minsize, flags=flags)

        for (x, y, w, h) in faces_coord:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(120,120,0),1)
            fram = gray[y:y+h,x:x+w]
            fram_col = frame[y:y+h,x:x+w]
            eyes = detector2.detectMultiScale(fram)
            for (e, f, g, k) in eyes:
                cv2.rectangle(fram_col,(e,f),(e+g,f+k),(140,100,0),1)
        cv2.imshow("Arya",frame)
        
        #code 27 is ESC key
        if cv2.waitKey(20) & 0xFF ==27:
            break



#camera release   
webcam.release()
