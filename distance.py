import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
import json

scaleRow = 13.99631528662423
scaleCol = 76.215
transform = np.float32([[ -3.36525864e-01,-2.38611758e+00,9.48230640e+02], [ -1.66679133e-02,-2.40999812e+00,8.27194741e+02], [ -2.81596111e-05,-3.30583419e-03,1.00000000e+00]])

horizon_row = 310
Y_f = 625.*2.5
camrea2origin = 2.5
########################read video and json########################
vc = cv2.VideoCapture('rec_20100101_005805.mp4')
c = 0
img=np.zeros(0)
if vc.isOpened():
    rval, img = vc.read()
    row, col = img.shape[0:2]
    host_loc = np.float32([int(col/2), row, 1]).T
    host_loc_trans = np.dot(transform,host_loc)
    temp_mat = host_loc_trans[2]
    host_loc_trans = host_loc_trans[[0,1]]
    host_loc_trans = (host_loc_trans/temp_mat).T
else:
    rval = False
    raise ValueError

#######################save video##################
fps = 25
fourcc = VideoWriter_fourcc(*"MJPG")
# videoWriter = cv2.VideoWriter("res_.avi", fourcc, fps, (col, row))
while rval:
    if not rval:
        break
    f = open("annotation/annotation_frame_%06d.json"%c, encoding='utf-8')
    test_json = json.load(f)
    c = c+1
    # if c<7000:
    #     rval, img = vc.read()
    #     continue
    objs = test_json['objects']
    imgCp = img.copy()
    cv2.circle(imgCp, tuple(host_loc_trans), 10, (0, 0, 255), -1)

    for labelInd in range(len(objs)):
        label = objs[labelInd]
        if label['score']<0.99:
            continue
        bbox = label['bbox']

        cv2.rectangle(imgCp,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[1][0]),int(bbox[1][1])),color=(0, 0, 255),thickness=2)
        cv2.putText(imgCp, '%s' % label['label'], (int(bbox[0][0]),int(bbox[0][1])),
                    color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        #########################algorithm 1 by perspective transform##############################
        remote_loc = np.float32([int((bbox[1][0]+bbox[0][0])/2), int(bbox[1][1]),1]).T
        remote_loc_trans = np.dot(transform,remote_loc)
        temp_mat = remote_loc_trans[2]
        remote_loc_trans = remote_loc_trans[[0,1]]
        remote_loc_trans = (remote_loc_trans/temp_mat).T
        dist1 = list(((host_loc_trans[0]-remote_loc_trans[0])/scaleCol,(host_loc_trans[1]-remote_loc_trans[1])/scaleRow))
        #########################algorithm 2 by perspective transform##############################
        pixeldist = abs(horizon_row - bbox[1][1])
        dist2 = Y_f / pixeldist
        dist2 = [dist1[0], dist2]

        cv2.putText(imgCp, 'x:%fm,y:%fm' % (dist1[0],dist1[1]), (int((bbox[1][0]+bbox[0][0])/2)-50, int(bbox[1][1]+10)),
                    color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        cv2.putText(imgCp, 'x:%fm,y:%fm' % (dist2[0],dist2[1]-camrea2origin), (int((bbox[1][0]+bbox[0][0])/2)-50, int(bbox[1][1]+25)),
                    color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        test_json['objects'][labelInd]['distance'] = [dist1[0]/scaleCol,dist1[1]/scaleRow]

    # ########################save json######################
    # with open("annotation_distance/annotation_frame_%06d.json"%(c-1), 'w') as json_file:
    #     json_file.write(json.dumps(test_json, indent=4))
    # videoWriter.write(imgCp)
    cv2.imshow('image',imgCp)
    cv2.waitKey(0)
    rval,img=vc.read()
# videoWriter.release()
vc.release()
cv2.destroyAllWindows()