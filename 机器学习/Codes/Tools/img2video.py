import os

import cap
import cv2

path="F:\\lpw\\ACAP\\objgan\\retest\\test3\\BtoA\\Apng"
out="F:\\lpw\\ACAP\\objgan\\retest\\test3\\BtoA"
list=os.listdir(path)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
vw = cv2.VideoWriter(out+"\\3.avi", fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
for i in list:
    frame = cv2.imread(path+"\\"+ i)
    cv2.waitKey(100)
    vw.write(frame)
vw.release()

