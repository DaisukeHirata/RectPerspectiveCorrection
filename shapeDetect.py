import math
import numpy as np
import cv2

#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 2
#camera
cap = cv2.VideoCapture(0)
print("press q to exit")
#correct perspective windows
_width  = 100.0*6
_height = 78.0*6
_margin = 0.0

corners = np.array(
    [
        [[  		_margin, _margin 			]],
        [[ 			_margin, _height + _margin  ]],
        [[ _width + _margin, _height + _margin  ]],
        [[ _width + _margin, _margin 			]],
    ]
)

pts_dst = np.array(corners, np.float32)

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

while(cap.isOpened()):
    #Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        #grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Canny
        canny = cv2.Canny(frame, 80, 240, 3)

        #contours
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0,len(contours)):
            #approximate the contour with accuracy proportional to
            #the contour perimeter
            approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

            #Skip small or non-convex objects
            if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                continue

            if(len(approx)>=4):
                #nb vertices of a polygonal curve
                vtc = len(approx)
                #get cos of all corners
                cos = []
                for j in range(2,vtc+1):
                    cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
                #sort ascending cos
                cos.sort()
                #get lowest and highest
                mincos = cos[0]
                maxcos = cos[-1]

                #Use the degrees obtained above and the number of vertices
                #to determine the shape of the contour
                x,y,w,h = cv2.boundingRect(contours[i])
                if(vtc==4):
                    cv2.putText(frame,'RECT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
                    cv2.drawContours(frame, [approx], -1, ( 255, 0, 0 ), 2 )
                    pts_src = np.array(approx, np.float32)
                    h, status = cv2.findHomography(pts_src, pts_dst)                    
                    warped = cv2.warpPerspective(frame, h, (int( _width + _margin * 2 ), int( _height + _margin * 2 )))
                    cv2.imshow('warped', warped)
                    break

        #Display the resulting frame
        #out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)
        if cv2.waitKey(1) == 1048689: #if q is pressed
            break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()