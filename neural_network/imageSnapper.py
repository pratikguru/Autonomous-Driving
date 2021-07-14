import pandas as pd
import os
import cv2
from datetime import datetime
from joystick import * 

global imgList, steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

#GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'ImagesCollection')
# print(myDirectory)

# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
        countFolder += 1
newPath = myDirectory +"/IMG"+str(countFolder)
os.makedirs(newPath)

# SAVE IMAGES IN THE FOLDER
def saveData(img,steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    steeringList.append(steering)

# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog():
    global imgList, steeringList
    rawData = {'Image': imgList,
                'Steering': steeringList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory,f'log_{str(countFolder)}.csv'), index=False, header=False)
    print('Log Saved')
    print('Total Images: ',len(imgList))


def returnFrameFromVideo(video):
  capture = cv2.VideoCapture(video)  
  return capture


if __name__ == '__main__':
    #cap = cv2.VideoCapture(0)

    cap = returnFrameFromVideo("test_video.mp4")

    while True:
        joyVal = getJS()
        steering = joyVal['axis1']
        throttle = joyVal['axis4']
        print("Steering: " + str(steering))
        ret, img = cap.read()
        if not ret:
          saveLog()  
          cv2.destroyAllWindows()
          break
        img = cv2.resize(img, (480, 240))
        saveData(img, steering)

        
        cv2.waitKey(1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    saveLog()  
    cap.release()
    cv2.destroyAllWindows()