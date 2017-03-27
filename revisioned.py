#!/usr/bin/python

import argparse
import imutils
import time
import cv2
import numpy as np

# GLOBALS
kernel5 = np.array([[0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0]]).astype(np.uint8)  # 5x5 convolution kernel, round-ish
maxObjects = 100  # how many objects to detect track at once
x = np.float32(np.zeros(maxObjects))  # x corner coordinates
y = np.float32(np.zeros(maxObjects))  # y corner coordinates
a = np.float32(np.zeros(maxObjects))  # object contour area
ao = np.float32(np.ones(maxObjects))  # area on last frame
xc = np.float32(np.zeros(maxObjects))  # x center coordinates
yc = np.float32(np.zeros(maxObjects))  # y center coordinates
w = np.float32(np.zeros(maxObjects))  # object width
h = np.float32(np.zeros(maxObjects))  # object height
xstart = np.float32(np.zeros(maxObjects))  # x position when object first tracked
xdist = np.float32(np.zeros(maxObjects))  # x distance travelled
xo = np.float32(np.zeros(maxObjects))  # x last frame center coordinates
xvel = np.float32(np.zeros(maxObjects))  # delta-x per frame
xvelFilt = np.float32(np.zeros(maxObjects))  # filtered delta-x per frame
yvelFilt = np.float32(np.zeros(maxObjects))  # filtered delta-y per frame
dArea = np.float32(np.zeros(maxObjects))  # change in enclosed area per frame
ystart = np.float32(np.zeros(maxObjects))  # y position when object first tracked
ydist = np.float32(np.zeros(maxObjects))  # y distance travelled
yo = np.float32(np.zeros(maxObjects))  # y last frame center coordinates
yvel = np.float32(np.zeros(maxObjects))  # delta-y per frame

# CONFIG VALUES TODO: move to json
procWidth = 620  # processing width (x resolution) of frame
fracF = 0.25  # adaptation fraction of background on each frame
GB = 15  # gaussian blur size
fracS = .25  # adaptation during motion event
noMotionCount = 0  # how many consecutive frames of no motion detected
motionCount = 0  # how many frames of consecutive motion
noMotionLimit = 2  # how many no-motion frames before start adapting
maxVel = 105  # fastest real-life reasonable velocity (not some glitch)
vfilt = 5  # stepwise velocity filter factor (low-pass filter)
xdistThresh = 2.5  # how many pixels an object must travel before it is counted as an event
ydistThresh = 2.5  # how many pixels an object must travel before it is counted as an event
xvelThresh = 1  # how fast object is moving along x before considered an event
yvelThresh = 1  # how fast object is moving along y before considered an event
yCropFrac = 10000  # crop off this (1/x) fraction of the top of frame (time/date string)
threshIter = 9  # count of iterations on the thresh stuff

# CONSTRUCT ARGUMENT PARSER
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# READ FROM VIDEO OR WEBCAM
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(4)
else:
    camera = cv2.VideoCapture(args["video"])

# INITIALIZE FIRST FRAME
averageFrame = None
slowFrame = None
motionDetect = False
minVal = 0
maxVal = 0
minLoc = -1
maxLoc = -1

# FIRST FRAME
(grabbed, frame) = camera.read()  # get very first frame

# Move on if we actually got a frame
if grabbed:
    frame = imutils.resize(frame, width=procWidth)  # resize to specified dimensions

    ysp, xsp, chan = frame.shape
    yspLim = 2 * ysp / 3  # allowable sum of heights of detected objects

    # make it gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # do the gaussian iteration
    gray = cv2.GaussianBlur(gray, (GB, GB), 0)  # Amount of Gaussian blur is a critical value

    averageFrame = gray
    slowFrame = gray

# END OF FIRST FRAME, START OF PROCESSING LOOP.
while grabbed:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    # resize frame
    frame = imutils.resize(frame, width=procWidth)
    # make it gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # do the gaussian iteration
    gray = cv2.GaussianBlur(gray, (GB, GB), 0)  # Amount of Gaussian blur is a critical value

    if noMotionCount > noMotionLimit:
        averageFrame = cv2.addWeighted(gray, fracF, averageFrame, (1.0 - fracF), 0)

    if motionCount == 1:  # reset current background to slowly-changing base background
        averageFrame = slowFrame

    if (noMotionCount > 30) and maxVal < 30:  # reset background when quiet, or too much
        averageFrame = gray  # hard reset to average filter; throw away older samples
        slowFrame = averageFrame
        noMotionCount = 0

    # calculate deltas
    frameDelta = cv2.absdiff(averageFrame, gray)  # difference of this frame from average background
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(frameDelta)

    # make the threshold
    thresh = cv2.dilate(thresh, kernel5, iterations=threshIter)  # join regions near each other
    # makes an array of contours
    (_, cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # variables initialised for the object detection iteration
    motionDetect = False  # we have not yet found motion in this frame
    motionStart = False  # haven't just started to detect a motion
    i = -1  # count of detected objects
    w = np.float32(np.zeros(maxObjects))  # reset object width
    h = np.float32(np.zeros(maxObjects))  # reset object height

    # Loop over all countours (possible objects);
    for c in cnts:
        # Contour to area
        area = cv2.contourArea(c)

        # Continue if the area is too small
        if area < args["min_area"]:
            continue

        # Create bounding rectangle to draw the object later on
        xt, yt, wt, ht = cv2.boundingRect(c)
        yct = yt + ht / 2

        i += 1  # We found a large enough object!

        # Make sure we don't go over our limit and crash stuff.
        if i >= maxObjects:  # ignore too-many objects
            continue

        # Save bounding box
        (x[i], y[i], w[i], h[i]) = (xt, yt, wt, ht)  # bounding box (x,y) and (width, height)

        # Save area
        a[i] = area
        dArea[i] = (1.0 * a[i]) / ao[i]  # ratio of current area to previous area

        # Calculate center coords of this contour
        xc[i] = x[i] + w[i] / 2
        yc[i] = y[i] + h[i] / 2

        # Calculate deltas
        xvel[i] = xc[i] - xo[i]  # delta-x since previous frame
        yvel[i] = yc[i] - yo[i]  # delta-x since previous frame

        # Calculate rolling average (x)
        if xvelFilt[i] == 0.0:  # is this the initial value?
            xvelFilt[i] = xvel[i]  # reset value without averaging
        else:
            xvelFilt[i] = (vfilt * xvel[i]) + (1.0 - vfilt) * xvelFilt[i]  # find the rolling average

        # Calculate rolling average (y)
        if yvelFilt[i] == 0.0:  # initial value?
            yvelFilt[i] = yvel[i]  # reset value without averaging
        else:
            yvelFilt[i] = (vfilt * yvel[i]) + (1.0 - vfilt) * yvelFilt[i]  # rolling average

        # Check whether we have a new object (big change detection)
        if (abs(xvel[i]) > maxVel) or (abs(yvel[i]) > maxVel) or dArea[i] > 2 or dArea[i] < 0.5:
            xvel[i] = 0
            yvel[i] = 0
            xstart[i] = xc[i]  # reset x starting point to 'here'
            ystart[i] = yc[i]  # reset x starting point to 'here'

        # Calculate distance the object has travelled
        xdist[i] = xc[i] - xstart[i]  # x distance this blob has travelled so far
        ydist[i] = yc[i] - ystart[i]  # y distance this blob has travelled so far

        # Save old coordinates
        xo[i] = xc[i]  # remember this coordinate for next frame
        yo[i] = yc[i]  # remember this coordinate for next frame
        ao[i] = a[i]  # remember old object bounding-contour area

        # When it is confirmed a real object
        if ((abs(xdist[i]) > xdistThresh) or (abs(ydist[i]) > ydistThresh)) \
                and ((abs(xvelFilt[i]) > xvelThresh) or (abs(yvelFilt[i] > yvelThresh))):
            print("%5.1f,%5.1f, %5.1f,  %5.2f, %5.0f" % (xc[i], ysp - yc[i], w[i], xvelFilt[i], xdist[i]))

            motionDetect = True
            # Draw rectangle around object
            cv2.rectangle(frame, (x[i], y[i]), (x[i] + w[i], y[i] + h[i]), (0, 0, 255), 2)

    # Do stuff when motion is detected
    if motionDetect:
        noMotionCount = 0
        motionCount += 1
    else:  # no motion found anywhere
        xvelFilt = np.float32(np.zeros(maxObjects))  # reset average motion to 0
        yvelFilt = np.float32(np.zeros(maxObjects))
        noMotionCount += 1
        motionCount = 0

    # Show original video with rectangles on it
    cv2.imshow("Video", frame)  # original video with detected rectangle and info overlay
    # Threshold output for debug
    cv2.imshow("Thresh", thresh)

    # KEY INPUTS
    key = cv2.waitKey(1) & 0xFF

    # Quit on 'esc' or q key
    if key == ord("q") or (key == 27):
        break
    if key == ord(" "):  # space to enter pause mode: wait until spacebar pressed again
        key = 0x00
        while key != ord(" "):
            key = cv2.waitKey(1) & 0xFF

camera.release()
cv2.destroyAllWindows()
