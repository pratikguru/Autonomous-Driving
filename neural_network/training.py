import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import random
from sklearn.model_selection import train_test_split 

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


def balanceSteeringData(data, display=True):
  nBin = 31
  samplesPerBin =  300
  hist, bins = np.histogram(data['Steering'], nBin)
  "Simple visualization of the steering angle. Categorization of the steering angles through the whole video."
  if display:
      center = (bins[:-1] + bins[1:]) * 0.5
      plt.bar(center, hist, width=0.03)
      plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
      plt.title('Data Visualisation')
      plt.xlabel('Steering Angle')
      plt.ylabel('No of Samples')
      plt.show()
  removeindexList = []
  for j in range(nBin):
      binDataList = []
      for i in range(len(data['Steering'])):
          if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
              binDataList.append(i)
      binDataList = shuffle(binDataList)
      binDataList = binDataList[samplesPerBin:]
      removeindexList.extend(binDataList)

  print('Removed Images:', len(removeindexList))
  data.drop(data.index[removeindexList], inplace=True)
  print('Remaining Images:', len(data))
  if display:
      hist, _ = np.histogram(data['Steering'], (nBin))
      plt.bar(center, hist, width=0.03)
      plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
      plt.title('Balanced Data')
      plt.xlabel('Steering Angle')
      plt.ylabel('No of Samples')
      plt.show()
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


def createModel():
  model = Sequential()

  model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
  model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
  model.add(Convolution2D(64, (3, 3), activation='elu'))
  model.add(Convolution2D(64, (3, 3), activation='elu'))

  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dense(1))

  model.compile(Adam(lr=0.0001),loss='mse')
  return model

def preProcess(img):
    img = img[54:120,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img 

def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering



def dataGen(imagesPath, steeringList, batchSize, trainFlag):
  while True:
      imgBatch = []
      steeringBatch = []

      for i in range(batchSize):
          index = random.randint(0, len(imagesPath) - 1)
          if trainFlag:
              img, steering = augmentImage(imagesPath[index], steeringList[index])
          else:
              img = mpimg.imread(imagesPath[index])
              steering = steeringList[index]
          img = preProcess(img)
          imgBatch.append(img)
          steeringBatch.append(steering)
      yield (np.asarray(imgBatch),np.asarray(steeringBatch))

#Initializating the data.
path = 'ImagesCollection'
data = importData(path)
print(data.head())


#Displaying steering angle graphs.
data = balanceSteeringData(data,display=True)


#Parsing data and reformatting data set.
#This splits the data from the steering and the image paths associated in the log files.
imagesPath, steerings = loadData(path, data)
print('No of Path Created for Images ',len(imagesPath),len(steerings))


#Split the data for testing, training and validation.
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)
print("Total training size: " + str(len(xTrain)))
print("Total validation size: " + str(len(xVal)))

#Fetching the model.
sampleModel = createModel()

#Train the model with our data.
history = sampleModel.fit(
  dataGen(xTrain, yTrain, 100, 1), 
  steps_per_epoch = 100, 
  epochs =10, 
  validation_data=dataGen(xVal, yVal, 50, 0),
  validation_steps = 50)

sampleModel.save('model.h5 ')
print("Model Saved")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
