import sys
sys.path.append('/home/orangepi/.local/lib/python3.8/site-packages')
import cv2
import time
import random
from rknnpool import rknnPoolExecutor
from func import myFunc

cap = cv2.VideoCapture(0)
                                                                                                           
modelPath = '/home/orangepi/Desktop/25_GCS_AM/25GCS_v5.rknn'
# 线程数, 增大可提高帧率
TPEs = 2
# 初始化rknn池
 
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)

# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()

# 初始化物体跟踪器
object_trackers = {}

while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    #print(frame.shape)
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    #print(frame.shape)
    cv2.namedWindow('yolov8', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('yolov8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        #print( 30 / (time.time() - loopTime))
        loopTime = time.time()

  
# 释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()
pool.release()
