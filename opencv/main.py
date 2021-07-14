import cv2
import numpy as np 
import urllib.request 

from utils import *

curveList = []
avgVal = 10

url ="http://192.168.0.178:8080/shot.jpg"
#cv2.namedWindow("screen", cv2.WINDOW_AUTOSIZE)

def returnFrame(url):
    imgResponse = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (480, 240))
    return img

def returnFrameFromVideo(video):
  capture = cv2.VideoCapture("test_video.mp4")  
  return capture

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


def getLaneCurve(img, display=2):
  
  imgCopy = img.copy()
  imgResult = img.copy()
  imgThres = thresholding(img)
  height, width, c = img.shape

  points = valTrackbars()
  imgWarp = warpImage(imgThres, points, width, height)
  imgWarpedWithTracePoints = drawTracePoints(imgCopy, points)

  midPoint, imgHist = getHistogram(imgWarp, display=True, minPer=0.5, region=4)
  curvedAveragePoint, imgHist = getHistogram(imgWarp, display=True, minPer=0.9)
  curveRaw = curvedAveragePoint - midPoint
  curveList.append(curveRaw)
  if len(curveList)>avgVal:
    curveList.pop(0)
  
  curve = int(sum(curveList)/len(curveList))

  if display != 0:
      imgInvWarp = warpImage(imgWarp, points, width, height,inv = True)
      imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
      imgInvWarp[0:height//3,0:width] = 0,0,0
      imgLaneColor = np.zeros_like(img)
      imgLaneColor[:] = 0, 255, 0
      imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
      imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
      midY = 450
      cv2.putText(imgResult,str(curve),(width//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
      cv2.line(imgResult,(width//2,midY),(width//2+(curve*3),midY),(255,0,255),5)
      cv2.line(imgResult, ((width // 2 + (curve * 3)), midY-25), (width // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
      for x in range(-30, 30):
          w = width // 20
          cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                  (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
      #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
      #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3)
      if display == 2:
        imgStacked = stackImages(0.7,([img,imgWarpedWithTracePoints,imgWarp],
                                        [imgHist,imgLaneColor,imgResult]))
        cv2.imshow('ImageStack',imgStacked)
      elif display == 1:
        cv2.imshow('Resutlt',imgResult)

  return curve


  

source = "video"
if __name__ == "__main__":
  
    initialTrackbarVals = [102, 80, 20, 214]
    initializeTrackbars(initialTrackbarVals)
    frameCounter = 0
    capture = returnFrameFromVideo("test_video.mp4")
    print("Capture Success!")
    #capture = returnFrame(url)

    while True:
      if source == "video":
        if capture.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
          capture.set(cv2.CAP_PROP_POS_FRAMES,0)
          frameCounter=0
        success, img = capture.read()
        img = cv2.resize(img, (480, 240))
      else:
        img = returnFrame(url)

      curveData = getLaneCurve(img, display=2)
      print(curveData)
      
      cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()
