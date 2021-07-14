from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt


steering_sensitivity = 1
max_throttle = 0.22
model = load_model("./model.h5")
print(model)


def pre_process_video_frame(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


def getName(filePath):
    myImagePathL = filePath.split('/')[-2:]
    myImagePath = os.path.join(myImagePathL[0],myImagePathL[1])
    return myImagePath

def importData(path):
    columns = ['Center', 'Steering']
    noOfFolders = len(os.listdir(path))//2
    data=pd.DataFrame()
    for x in range(0, 5):
      dataNew = pd.read_csv(os.path.join(path,f'log_{x}.csv'), names=columns)
      print(f'{x}:{dataNew.shape[0]} ',end='')
      dataNew['Center']=dataNew['Center'].apply(getName)
      data = data.append(dataNew,True )
    print(" ")
    print("Total images imported: " + str(data.shape[0]))
    return data


def loadData(path, data):
  imagesPath = []
  steering = []
  for x in range(len(data)):
    indexed_data = data.iloc[x]
    imagesPath.append( os.path.join(path, indexed_data[0]) )
    steering.append(float(indexed_data[1]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering

def returnFrameFromVideo(video):
    capture = cv2.VideoCapture(video)  
    return capture

def averageSteeringValue(rotations, img):
  buffer = []
  for x in range(0, rotations):
    buffer.append(float(model.predict(img)))
  return sum(buffer)/rotations

if __name__ == "__main__":
    cap = returnFrameFromVideo("test_video.mp4")
    counter = 0 
    path = 'ImagesCollection'
    data = importData(path)
    print(data.head())

    imagesPath, steerings = loadData(path, data)
    print('No of Path Created for Images ',len(imagesPath),len(steerings))
    frame_counter = 0
    frame_buffer = []
    while True:
      counter = counter + 1
      ret, img = cap.read()
      if not ret:
        print("Loop Complete")
        break
      else:
        img = cv2.resize(img, (480, 240))
        img = np.asarray(img)
        img = pre_process_video_frame(img)
        img = np.array([img])
        frame_counter += 1
        start_time = time.clock()
        #steering = float(model.predict(img))
        steering = averageSteeringValue(10, img)
        end_time = time.clock() - start_time 
        frame_buffer.append(end_time)
        print("----------------------------------")
        print("Time: " + str(end_time))
        print("Frame: " + str(frame_counter))
        print("Curve Value: " + str(steering))
        print("----------------------------------")
        plt.plot(frame_counter, steering * steering_sensitivity, '-o')
        plt.xlabel("Frame Count")
        plt.ylabel("Steering Values")
        plt.title("The steering value of every frame.")
        plt.pause(0.00000001)


        plt.plot()
        cv2.waitKey(1)
    
    print("Average Time: " + str(sum(frame_buffer)/frame_counter))
    print("Total Time: " + str(sum(frame_buffer)))
    print("Total Frames: " + str(frame_counter))