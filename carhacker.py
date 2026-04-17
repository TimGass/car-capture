import api
import cv2
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
import numpy as np
import imutils
import dlib
import argparse
import time
import threading
from flask import Flask, render_template, Response
from flask_cors import CORS
import socketio
import queue

sio = socketio.Server(async_mode="threading", cors_allowed_origins="*", host="0.0.0.0", port=5000)
app = Flask(__name__)
CORS(app)
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)

vs = None
input = None
t = None

subscribers = []

def gen_frames(prototxt:str='./mobilenet_ssd/MobileNetSSD_deploy.prototxt', model:str='./mobilenet_ssd/MobileNetSSD_deploy.caffemodel', output:str=None, confidence:float=0.4, skip_frames:int=2):
    global outputFrame, lock, vs, input
    list_of_garages = api.get_garages()
    garage = next(garage for garage in list_of_garages if garage.name == 'Johnson')
    print(garage.name)

    available_spaces = garage.capacity - garage.cars_in_lot

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=10, maxDistance=20)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalIn = 0
    totalOut = 0
    
    startTime = time.time()
    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        if input:
          ret, frame = vs.read()
        else:
          frame
        if not ret and input is not None:
            totalFrames = 0
            trackers = []
            trackableObjects = {}
            vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = frame if input else ret
        
        if time.time() - startTime > 300:
            startTime = time.time()
            totalFrames = 0
            totalIn = 0
            totalOut = 0
            trackers = []
            trackableObjects = {}
            ct = CentroidTracker(maxDisappeared=10, maxDistance=20)
            vs.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 30,
                                    (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                temp_confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if temp_confidence > confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a car, ignore it
                    if (CLASSES[idx] != "car"):
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)
        # loop over the trackers
        else:
          for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)
        
        if totalFrames > 20:
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < H // 2:
                            totalOut += 1
                            to.counted = True
                            if available_spaces < garage.capacity:
                                garage.cars_in_lot -= 1
                                available_spaces += 1
                                # api.put_garage(garage)

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[1] > H // 2:
                            totalIn += 1
                            to.counted = True
                            if garage.cars_in_lot < garage.capacity:
                                garage.cars_in_lot += 1
                                available_spaces -= 1
                                # api.put_garage(garage)
                            elif garage.cars_in_lot == garage.capacity:
                                garage.cars_in_lot = 1
                                available_spaces = garage.capacity - 1
                                # api.put_garage(garage)
                                

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Out", totalOut),
            ("In", totalIn),
            ("Available", available_spaces),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)
            
        for client in subscribers:
            outputFrame = frame.copy()
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if flag:
                client.put(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')
        # increment the total number of frames processed thus far
        totalFrames += 1
        

# Uncomment for flask server integration
# def generate():
#   # grab global references to the output frame and lock variables
#   global outputFrame, lock
#   # loop over frames from the output stream
#   while True:
#     # wait until the lock is acquired
#     with lock:
#       # check if the output frame is available, otherwise skip
#       # the iteration of the loop
#       if outputFrame is None:
#         continue
#       # encode the frame in JPEG format
#       (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
#       # ensure the frame was successfully encoded
#       if not flag:
#         continue
#     # yield the output frame in the byte format
#     yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
#       bytearray(encodedImage) + b'\r\n')

# @app.route('/')
# def index():
#   return render_template('index.html') 

@app.route('/video_feed')
def video_feed():
  try:
      yolo = queue.Queue(maxsize=10)
      subscribers.append(yolo)
      def popQueue():
          while True:
              try:
                  yield yolo.get_nowait()
              except queue.Empty:
                  yield b''
  except GeneratorExit:
      print("Client disconnected, stopping process")
      subscribers.remove(yolo)
  return Response(popQueue(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
if __name__ == "__main__":
  # construct the argument parser and parse command line arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i","--ip", type=str, default='localhost',
    help="host address")
  ap.add_argument("-p", "--port", type=int, default=5000,
    help="port number")
  ap.add_argument("-v", "--video-file", type=str, default='./videos/cars_real.mp4',
    help="input file for the video to be used, set to none for camera")
  args = vars(ap.parse_args())
  input = args['video_file']
  # if a video path was not supplied, grab a reference to the webcam
  if not input:
      print("[INFO] starting video stream...")
      vs = VideoStream(src=0).start()

  # otherwise, grab a reference to the video file
  else:
      print("[INFO] opening video file...")
      vs = cv2.VideoCapture(input)
  t = threading.Thread(target=gen_frames)
  t.daemon = True
  t.start()
  app.run(threaded=True, host='0.0.0.0', port=5000)
  

if vs is not None:
  if input:
    vs.release()
  else:
    vs.stop()