# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

import cv2 as cv
import argparse
import sys
import tempfile
import numpy as np
import os.path
from typing import List
import logging
import os
import json
import shutil

log = logging.getLogger()


def blurRegion(frame, x, y, w, h):
    """
    Blur https://stackoverflow.com/questions/24195138/gaussian-blurring-with-opencv-only-blurring-a-subregion-of-an-image
    """

    # Grab ROI with Numpy slicing and blur
    ROI = frame[y:y + h, x:x + w]
    blur = cv.GaussianBlur(ROI, (51, 51), 0)

    # Insert ROI back into image
    frame[y:y + h, x:x + w] = blur


def getOutputsNames(net):
    """
    Get the names of the output layers
    """
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame, left, top, right, bottom):
    """
    Draw the predicted bounding box
    """
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)


def postprocess(frame, outs, confThreshold, nmsThreshold):
    """
    Remove the bounding boxes with low confidence using non-maxima suppression
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                log.info(detection[4], " - ", scores[classId],
                         " - th : ", confThreshold)
                log.info(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # drawPred(frame, left, top, left + width, top + height)
        blurRegion(frame, left, top, width, height)


def process(video_path, dst, model_path):
    if not os.path.isfile(video_path):
        raise Exception("file %s not found" % video_path)

    log.info("start process")
    # Initialize the parameters
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold

    inpWidth = 416  # 608     # Width of network's input image
    inpHeight = 416  # 608     # Height of network's input image

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "darknet-yolov3.cfg")
    outputFile = "result.avi"

    net = cv.dnn.readNetFromDarknet(modelConfiguration, model_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    cap = cv.VideoCapture(video_path)

    # Get the video writer initialized to save the output video
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))  # TODO Upload

    while True:
        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            log.info("Done processing. Output file is stored as %s" %
                     outputFile)
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(
            frame,
            outs,
            confThreshold,
            nmsThreshold
        )

        # Write the frame with the detection boxes
        vid_writer.write(frame.astype(np.uint8))
