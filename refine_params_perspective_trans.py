import numpy as np

gt = [
    {
        'bbox' : [[894,264],[1198,484]],
        'distance' : [-2.4,6.6]
    },
    {
        'bbox' : [[814,285],[920,360]],
        'distance' : [0,25.2]
    },
    {
        'bbox' : [[691,283],[834,398]],
        'distance' : [0,15.5]
    },
    {
        'bbox' : [[693,286],[771,356]],
        'distance' : [0,31.4]
    },
    {
        'bbox' : [[528,245],[878,531]],
        'distance' : [0,4.2]
    },
    {
        'bbox' : [[865,222],[1225,496]],
        'distance' : [-2.5,5.9]
    },
    {
        'bbox' : [[591,251],[880,503]],
        'distance' : [0,5.9]
    },
    {
        'bbox' : [[6,267],[268,486]],
        'distance' : [5,5.9]
    }
]

scaleRow = 14.70472184748387#error 5.65320987503
scaleRow = 13.6972#error 6.98771836052
scaleRow = 15.92492021755556#error 8.6965323901
scaleRow = 12.84810825044068#error 13.3121318264
scaleRow = 13.9968152866242#error 5.56486356733
scaleRow = 13.99631528662423#error 5.56479800949(best)
scaleCol = 96.215
transform = np.float32([[ -3.36525864e-01,-2.38611758e+00,9.48230640e+02], [ -1.66679133e-02,-2.40999812e+00,8.27194741e+02], [ -2.81596111e-05,-3.30583419e-03,1.00000000e+00]])

row = 720
col = 1280
host_loc = np.float32([int(col/2), row, 1]).T
host_loc_trans = np.dot(transform,host_loc)
temp_mat = host_loc_trans[2]
host_loc_trans = host_loc_trans[[0,1]]
host_loc_trans = (host_loc_trans/temp_mat).T

flag = 0
step = 0.01
refineRange = 10000
if flag == 0:
    refineScaleRowOrCol = scaleCol - (refineRange*step)/2
else:
    refineScaleRowOrCol = scaleRow - (refineRange*step)/2

minError = 9999999
for index in range(refineRange):
    refineScaleRowOrCol = refineScaleRowOrCol+step
    errorSum = 0
    for indexGT in range(len(gt)):
        bbox = gt[indexGT]['bbox']
        distance_gt = gt[indexGT]['distance']

        #########################algorithm 1 by perspective transform##############################
        remote_loc = np.float32([int((bbox[1][0] + bbox[0][0]) / 2), int(bbox[1][1]), 1]).T
        remote_loc_trans = np.dot(transform, remote_loc)
        temp_value = remote_loc_trans[2]
        remote_loc_trans = remote_loc_trans[[0, 1]]
        remote_loc_trans = (remote_loc_trans / temp_value).T

        ########################error############################################
        if flag==0:
            dist1 = (host_loc_trans[0] - remote_loc_trans[0]) / refineScaleRowOrCol
            errorSum = errorSum + abs(distance_gt[0]-dist1)
        else:
            dist1 = (host_loc_trans[1] - remote_loc_trans[1]) / refineScaleRowOrCol
            errorSum = errorSum + abs(distance_gt[1]-dist1)
    if minError>errorSum:
        minError = errorSum
        bestRefineScaleRowOrCol = refineScaleRowOrCol
    print(errorSum,'\n',minError,'\n',refineScaleRowOrCol,'\n')
print(minError,bestRefineScaleRowOrCol)
