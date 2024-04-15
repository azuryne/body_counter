from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from tracker.centroidtracker import CentroidTracker 
from tracker.trackableobject import TrackableObject

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,help="video footage CCTV address")
ap.add_argument("-o", "--output", type=str, help="Results counter people address") 
ap.add_argument("-c", "--confidence", type=float, default=0.8, help="minimum possibility to detect human body")
ap.add_argument("-s", "--skip-frames", type=int, default=80,help="# of skip frames every object detection process")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

# Read model or load model
net = cv2.dnn.readNetFromCaffe("D:\\SelfStudy\\ai_dev\\Meeting1415\\mobilenet_ssd\\MobileNetSSD_deploy.prototxt", 
                               "D:\\SelfStudy\\ai_dev\\Meeting1415\\mobilenet_ssd\\MobileNetSSD_deploy.caffemodel")

if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    
fps = FPS().start()

writer = None
W = None
H = None
ct = CentroidTracker(maxDisappeared=77, maxDistance=80)
# Add code below
print(ct.objects, ct.disappeared,ct.maxDisappeared, ct.maxDistance)
trackers = []
trackableObjects = {}
totalFrames = 0
totalBodies = 0  

# Add under the code FPS
while True:
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame
    if args["input"] is not None and frame is None:
        break

    frame = imutils.resize(frame, width=1000)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if W is None or H is None:
        print("frame.shape", frame.shape)
        (H, W) = frame.shape[:2]
        
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG") # determine the format
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    status = "Waiting"
    rects = []
        
    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        print("Detecting........")
        # trackers = []
        # trackableObjects = {}
        # totalFrames = 0
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W,H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in np.arange(0, detections.shape[2]):
            print(detections.shape[2])
            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
        print(trackers)
        
    else: 
        for tracker in trackers:
            print("Tracking.......")
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))
        print(rects)
            
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None: 
            to = TrackableObject(objectID, centroid)
        else:
            to.centroids.append(centroid)
            if not to.counted:
                totalBodies += 1
                to.counted = True
        trackableObjects[objectID] = to 
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # cv2.putText(frame, f"Visitors : {totalBodies}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.putText(frame, f"Status : {status}", (10, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
    if writer is not None:
        writer.write(frame)
    totalFrames += 1
    fps.update()
        # print(totalFrames)
    cv2.putText(frame, f"Visitors : {totalBodies}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Status : {status}", (10, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
fps.stop()
# To close VideoStream from Webcam
if not args.get("input", False):
    vs.stop()
# To close from internal file Video input
else:
    vs.release()
cv2.destroyAllWindows()