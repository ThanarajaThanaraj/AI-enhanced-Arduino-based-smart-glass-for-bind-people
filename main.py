import cv2 as cv 
import numpy as np
import pyttsx3
import face_recognition
import urllib.request
import time
import os

url='http://192.168.67.239/cam-hi.jpg'

engine = pyttsx3.init('sapi5')           
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

path = 'dataset'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" %(class_names[classid[0]], score)
        la = class_names[classid[0]]
        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
        print(la)
        if la =='person':
            speak("PERSON HERE")  
        else:
            speak(la)      
        # time.sleep(1)
       
        if classid ==67: # cell phone
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        # adding more classes for distnaces estimation 
        elif classid ==2: # car
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==15: # cat
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==16: # dog
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==14: # bird
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==17: # horse
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==18: # sheep
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==19: # cow
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==20: # elephant
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==21: # bear
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==22: # zebra
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==23: # girafe
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==25: # umbrella
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==26: # handbag
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==32: # sports ball
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==39: # bottle
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==40: # wine glass
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==41: # cup
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==42: # fork
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==43: # knife
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==44: # spoon
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==45: # bowl
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==46: # banana
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==47: # apple
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==48: # sandwich
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==49: # orange
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==51: # carrot
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==55: # cake
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==56: # chair
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==63: # laptop
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==64: # mouse
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==65: # remote
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==66: # keyboard
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==73: # book
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==74: # clock
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==76: # scissors
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        
    return data_list



# cap = cv.VideoCapture(0)
counts = 0
while True:
    # ret, frame = cap.read()
    # imgS = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    # imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv.imdecode(imgnp,-1)
    imgS = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)
    # frame = frame.rotate(90, expand=True)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    data = object_detector(frame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2-35), (x2, y2), (0, 250, 0), cv.FILLED)
            cv.putText(frame, name, (x1+6, y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if (name =='DHANARAJ1') or (name =='DHANARAJ2') or (name =='DHANARAJ3'):
                print(" DHANARAJ HERE")
                speak("DHANARAJ Here")
            # elif (name =='DHANARAJ1') or (name =='DHANARAJ2') or (name =='DHANARAJ3'):
            #     print(" DHANARAJ HERE")
            #     speak("DHANARAJ Here")
            # elif (name =='DINESH'):
            #     print("DINESH HERE")
            #     speak("DINESH Here")
            #elif (name =='john') or (name =='john1'):
               # print("john here")
                #speak("john here")
            elif(name =='john'):
                print("john HERE")
               speak("john Here")

    
    cv.imshow('Blind People Helping Application',frame)
    
    # cv.waitKey(1)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()

