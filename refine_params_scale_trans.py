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

horizon_row = 300
Y_f = 625.
camrea2origin = 2.5#distance from camera to the start measure point(m)

row = 720
col = 1280

minError = 9999999
for indexHori in range(100):
    mult_scale = 0.6
    horizon_row_temp = horizon_row - 50 + indexHori
    for indexY_f in range(60):
        mult_scale = mult_scale + 0.1
        Y_f_temp = Y_f * mult_scale
        error2y = 0
        for indexGT in range(len(gt)):#len(gt)
            # indexGT = indexGT+index_temp
            bbox = gt[indexGT]['bbox']

            #########################algorithm 2 by perspective transform##############################
            pixeldist = abs(horizon_row_temp - bbox[1][1])
            dist2 = Y_f_temp / pixeldist - camrea2origin

            ########################error############################################
            distance_gt = gt[indexGT]['distance']
            error2y = error2y + abs(distance_gt[1]-dist2)
        if minError>error2y:
            minError = error2y
            minHorizon_row_temp = horizon_row_temp
            minMult_scale = mult_scale
        print(error2y,'\n',minError,'\n',horizon_row_temp,mult_scale,'\n','\n\n')
print(minHorizon_row_temp,minMult_scale)
