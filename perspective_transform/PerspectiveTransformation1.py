import cv2
import numpy as np

imgCount = 0
imgPoints = []
objCount = 0
objPoints = []
cv2.namedWindow('src')

def get_points(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        global imgCount
        if imgCount<4:
            imgPoints.append([x,y])
            cv2.circle(originalImage, (x, y), 2, (255, 0, 0), -1)
            cv2.imshow('src', originalImage)
            imgCount = imgCount + 1
    elif event==cv2.EVENT_RBUTTONDOWN:
        global objCount
        if objCount<2:
            objPoints.append([x,y])
            cv2.circle(originalImage, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow('src', originalImage)
            objCount = objCount + 1
        else:
            cv2.rectangle(originalImage,tuple(objPoints[0]),tuple(objPoints[1]),(0, 0, 255),2)
            cv2.imshow('src', originalImage)

cv2.setMouseCallback('src', get_points)

Image = cv2.imread("2200.png")
w, h = Image.shape[0:2]
scale = 1
originalImage=cv2.resize(Image,(int(h/scale), int(w/scale)),interpolation=cv2.INTER_CUBIC)
w, h = originalImage.shape[0:2]
originalImageCp = originalImage.copy()
cv2.imshow('src',originalImage)
cv2.waitKey(0)

imagePoints = np.array(0)
if imgCount>=4:
    imagePoints=np.float32((imgPoints))
    imagePoints=np.float32(([615,365],[784,364],[1040,619],[245,616]))
    cv2.line(originalImage, tuple(imagePoints[0]), tuple(imagePoints[1]), (0, 255, 0), 2)
    cv2.line(originalImage, tuple(imagePoints[1]), tuple(imagePoints[2]), (0, 255, 0), 2)
    cv2.line(originalImage, tuple(imagePoints[2]), tuple(imagePoints[3]), (0, 255, 0), 2)
    cv2.line(originalImage, tuple(imagePoints[3]), tuple(imagePoints[0]), (0, 255, 0), 2)

moveValueX = 0.0
moveValueY = 0.0
objectivePoints = np.array(0)
if objCount>=2:
    objectivePoints = np.float32(([objPoints[0][0]+moveValueX,objPoints[0][1]+moveValueY],[objPoints[1][0]+moveValueX,objPoints[0][1]+moveValueY],
                            [objPoints[1][0]+moveValueX, objPoints[1][1]+moveValueY],[objPoints[0][0]+moveValueX,objPoints[1][1]+moveValueY]))
    objectivePoints = np.float32(([324+moveValueX,138+moveValueY],[938+moveValueX,138+moveValueY],[938+moveValueX,622+moveValueY],[324+moveValueX,622+moveValueY]))
    objectivePoints = np.float32(([579 + moveValueX, 280 + moveValueY], [817 + moveValueX, 280 + moveValueY],
                              [817 + moveValueX, 634 + moveValueY], [579 + moveValueX, 634 + moveValueY]))
# [100.0/scale + moveValueY, 146.0/scale + moveValueX],[920.0/scale + moveValueY, 146.0/scale + moveValueX],
#                          [920.0/scale + moveValueY, 700.0/scale + moveValueX],[100.0/scale + moveValueY,700.0/scale + moveValueX]
    cv2.line(originalImage, tuple(objectivePoints[0]), tuple(objectivePoints[1]), (0, 255, 0), 2)
    cv2.imshow("src", originalImage)
    cv2.waitKey(0)
    cv2.line(originalImage, tuple(objectivePoints[1]), tuple(objectivePoints[2]), (0, 255, 0), 2)
    cv2.imshow("src", originalImage)
    cv2.waitKey(0)
    cv2.line(originalImage, tuple(objectivePoints[2]), tuple(objectivePoints[3]), (0, 255, 0), 2)
    cv2.imshow("src", originalImage)
    cv2.waitKey(0)
    cv2.line(originalImage, tuple(objectivePoints[3]), tuple(objectivePoints[0]), (0, 255, 0), 2)
    cv2.imshow("src", originalImage)
    cv2.waitKey(0)

if imgCount>=4 and objCount>=2:
    transform = cv2.getPerspectiveTransform(imagePoints, objectivePoints)
    ########test Mat#############
    b = np.ones(4)
    imagePoints = np.column_stack((imagePoints,b))
    testMat = (np.dot(transform,imagePoints.T)).T
    tempMat = testMat[:,2]
    print(tempMat.shape)
    testMat = testMat[:,[0,1]]
    testMat = testMat.T/tempMat.T
    print(imagePoints, objectivePoints, transform, testMat.T)
else:
    transform = np.float32([[ -5.91116069e-01,-2.40815286e+00,1.16978301e+03],[ -6.28356528e-02,-2.50024159e+00,9.24861461e+02],[ -1.02848871e-04,-3.06797026e-03,1.00000000e+00]])
    transform = np.float32([[ -3.36525864e-01,-2.38611758e+00,9.48230640e+02], [ -1.66679133e-02,-2.40999812e+00,8.27194741e+02], [ -2.81596111e-05,-3.30583419e-03,1.00000000e+00]])
perspectiveImage = cv2.warpPerspective(originalImageCp,transform, (int(h*scale),int(w*scale)), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


cv2.imshow("src", originalImage)
cv2.imshow("res", perspectiveImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
