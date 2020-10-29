import os
import tflite_runtime as tflite
from tflite_runtime.interpreter import Interpreter
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


class VideoStream(object):
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture()
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='Name of the .tflite file',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Min confidence threshold to display objects',
                        default=0.5)
    parser.add_argument('--resolution', help='WxH resolution',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU accel',
                        action='store_true')
    return parser


def main():
    parser = generate_parser()
    args = parser.parse_args()

    PATH_TO_CKPT = os.path.abspath(args.graph)
    PATH_TO_LABELS = os.path.abspath(args.labels)
    MIN_THRESHOLD = float(args.threshold)
    RES_W, RES_H = args.resolution.split('x')
    IMAGE_W, IMAGE_H = int(RES_W), int(RES_H)
    USE_TPU = args.edgetpu

    # Load label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    INPUT_MEAN = 127.5
    INPUT_STD = 127.5

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    video_stream = VideoStream(resolution=(IMAGE_W, IMAGE_W), framerate=30)
    video_stream.start()
    time.sleep(1)

    while True:

        # Start timer
        timer_start = cv2.getTickCount()

        # grab frame from video stream
        frame_start = video_stream.read()
        
        # Acquire fram and resize to expected shape [1xHxWx3]
        frame = frame_start.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLO_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model
        if floating_model:
            input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD

        # Perform the actual detencion by running the model width image
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
        for i in range(len(scores)):
            if ((scores[i] > MIN_THRESHOLD) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                ymin = int(max(1, (boxes[i][0] * IMAGE_H)))
                xmin = int(max(1, (boxes[i][1] * IMAGE_W)))
                ymax = int(min(IMAGE_H, (boxes[i][2] * IMAGE_H)))
                xmax = int(min(IMAGE_W, (boxes[i][3] * IMAGE_W)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    
                # Draw label
                object_name = labels[int(classes[i])]
                label = f"{object_name}: {scores[i]*100}%"
                label_size, base_line = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(
                        frame,
                        (xmin, label_ymin-label_size[1]-10),
                        (xmin+label_size[0], label_ymin+base_line-10),
                        (255, 255, 255),
                        cv2.FILLED
                )
                cv2.putText(
                        img=frame,
                        text=label,
                        org=(xmin, label_ymin-7),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 0, 0),
                        thickness=2
                )

        # Draw framerate in corner of frame
        cv2.putText(
            frame,
            f"FPS: {frame_rate_calc:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        cv2.imshow('Object detector', frame)

        timer_end = cv2.getTickCount()
        time_ = (timer_end - timer_start)/freq
        frame_rate_calc = 1 / time_

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # clean up
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == '__main__':
    main()
