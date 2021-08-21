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
from awscli.clidriver import create_clidriver

log = logging.getLogger()


def aws_cli(envs: dict, cmd: List[str]):
    old_env = dict(os.environ)
    try:
        # Environment
        env = os.environ.copy()
        env['LC_CTYPE'] = u'en_US.UTF'
        for k, v in envs.items():
            env[k] = v
        os.environ.update(env)

        # Run awscli in the same process
        exit_code = create_clidriver().main(args=cmd)

        # Deal with problems
        if exit_code > 0:
            raise RuntimeError('AWS CLI exited with code {}'.format(exit_code))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


class DownloadManager:
    def __init__(self, bucket_name, region_name, aws_access_key_id, aws_secret_access_key):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        self.envs = {
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            "AWS_DEFAULT_REGION": self.region_name
        }

    def download(self, src, dst):
        extension = os.path.splitext(src)[1]
        if extension == "":
            # is a directory
            action = "sync"
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
        else:
            action = "cp"
        src = "s3://%s/%s" % (self.bucket_name, src.lstrip('/'))
        dst = os.path.abspath(dst)
        log.info("download", extra={"src": src, "dst": dst})
        aws_cli(self.envs, ['s3', action, src, dst])

    def upload(self, src, dst, is_public=False):
        src = os.path.abspath(src)
        action = "cp" if os.path.isfile(src) else "sync"
        dst = "s3://%s/%s" % (self.bucket_name, dst.lstrip('/'))
        log.info("upload", extra={"src": src, "dst": dst})
        args = ['s3', action, src, dst]
        if is_public:
            args = args + ['--acl', 'public-read']
        aws_cli(self.envs, args)


download_manager = DownloadManager(
    bucket_name="hack0820",
    aws_access_key_id="AKIAU7XZOMYFZK54A5UD",
    aws_secret_access_key="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",
    region_name="eu-central-1",
)

# Blur https://stackoverflow.com/questions/24195138/gaussian-blurring-with-opencv-only-blurring-a-subregion-of-an-image


def blurRegion(x, y, w, h):
    # Grab ROI with Numpy slicing and blur
    ROI = frame[y:y + h, x:x + w]
    blur = cv.GaussianBlur(ROI, (51, 51), 0)

    # Insert ROI back into image
    frame[y:y + h, x:x + w] = blur

# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box


def drawPred(left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

# Remove the bounding boxes with low confidence using non-maxima suppression


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                print(detection[4], " - ", scores[classId],
                      " - th : ", confThreshold)
                print(detection)
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
        # drawPred(left, top, left + width, top + height)
        blurRegion(left, top, width, height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--inputVideoPath', help='Path to video file.')
    args = parser.parse_args()

    model_path = "./model.weights"
    if not os.path.isfile(model_path):
        download_manager.download("/model.weights", model_path)

    target_path = tempfile.mkdtemp()
    download_manager.download(args.inputVideoPath, target_path)
    video_path = "%s/%s" % (target_path, args.inputVideoPath)

    # Initialize the parameters
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold

    inpWidth = 416  # 608     # Width of network's input image
    inpHeight = 416  # 608     # Height of network's input image

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "darknet-yolov3.cfg"
    modelWeights = model_path
    outputFile = "result.avi"

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    if not os.path.isfile(video_path):
        print("Input video file ", video_path, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(video_path)

    # Get the video writer initialized to save the output video
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))  # TODO Upload

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing.")
            print("Output file is stored as ", outputFile)
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
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (
            t * 1000.0 / cv.getTickFrequency())

        # Write the frame with the detection boxes
        vid_writer.write(frame.astype(np.uint8))

    download_manager.upload(outputFile, "result_%s" % args.inputVideoPath)
