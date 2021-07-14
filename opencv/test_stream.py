import cv2
import urllib.request 
import numpy as np 





url ="http://192.168.0.178:8080/shot.jpg"
#cv2.namedWindow("screen", cv2.WINDOW_AUTOSIZE)

def returnFrame(url):
    imgResponse = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (480, 240))
    return img

while True:

    #cv2.imshow("screen", returnFrame(url))
    print(returnFrame(url))
    key = cv2.waitKey(5)
    if key == ord('q'):
        break 
    

cv2.destroyAllWindows