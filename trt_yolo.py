"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import ctypes
# import subprocess
import threading
# from tqdm import tqdm

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'
start_time = 0
start_line = 0

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-p', '--perf', action='store_true',
        help='enable performance(overall execution time) profiling')
    parser.add_argument(
        '-w', '--watts', action='store_true',
        help='enable power consumption profiling')
    parser.add_argument(
        '-f', '--freq', type=int,
        help='specify the initial GPU frequency index (0 ~ 11)')
    args = parser.parse_args()
    return args

def run_daemon(lib):
    lib.power_monitoring_daemon.argtypes = [ctypes.c_bool]
    lib.power_monitoring_daemon(True)
# 100 gcc
# PID Controller Variables
Kp = 1.8 # Proportional gain
Ki = 0.001  # Integral gain
Kd = 0.05  # Derivative gain

integral_error = 0
previous_error = 0

recent_boxes = []
boxes_entries = 10
recent_perfs = []
perf_entries = 1000

# Adaptively choose #bounding box as target
def update_boxex_target(curr_boxes):
    recent_boxes.append(curr_boxes)
    if len(recent_boxes) > boxes_entries:
        recent_boxes.pop(0)
    target_boxes = sum(recent_boxes) / len(recent_boxes)
    return target_boxes

# Adaptively choose performance target
def update_adaptive_target(current_processing_time):
    recent_perfs.append(current_processing_time)
    if len(recent_perfs) > perf_entries:
        recent_perfs.pop(0)
    adaptive_target = sum(recent_perfs) / len(recent_perfs)
    return adaptive_target

def energy_per_frame(start_line):
    energy = 0.0
    with open('power_consump.txt', 'r') as file:
        file.seek(start_line)
        for line in file:
            power, _ = line.strip().split('\t')
            power = float(power)

            energy += power * 0.01
        end_line = file.tell()

    return energy, end_line


def pid_update(current_value, target_value, dt):
    global integral_error, previous_error
    error = current_value - target_value

    # Proportional term
    P_out = Kp * error
    # Integral term
    integral_error += error * dt
    I_out = Ki * integral_error
    # Derivative term
    derivative = (error - previous_error) / dt
    D_out = Kd * derivative
    # Total output
    total_output = P_out + I_out + D_out
    # Update previous error for next iteration
    previous_error = error

    return total_output

def normalize_output(pid_output, min_output, max_output):
    return (pid_output - min_output) / (max_output - min_output)

def calculate_bbox_adjustment(curr_boxes):
    # Calculates a scale factor based on the difference between curr_boxes and target_boxes
    max_boxes = 20
    print(f"curr_boxes Output: {curr_boxes}")

    curr_boxes = min(curr_boxes, max_boxes)
    scale_factor = curr_boxes / max_boxes

    return scale_factor

def calculate_gpu_utilization_adjustment(gpu_util):
    # gpu_util will be 0~1000, return 0~1 as the intent to scale up frequency
    normalized_util = gpu_util / 1000.0
    print(f"gpu_util Output: {gpu_util}")
    scale_factor = (1 - normalized_util) ** 2
    return scale_factor

def combine_adjustments(perf_adj, bbox_adj, gpu_adj):
    print(f"perf Output: {perf_adj}")
    # print(f"box Output: {bbox_adj}")
    # print(f"gpu Output: {gpu_adj}")
    weight_perf = 1.0
    weight_bbox = 0.0
    weight_gpu = 0.0
    return weight_perf * perf_adj + weight_bbox * bbox_adj - weight_gpu * gpu_adj

# Function to map the adjustment value to index diff
def new_freq_index(adjustment):
    # choose between up or down within 10 frequency index steps
    index = adjustment * 10
    return index
    # if adjustment < 0.5:
    #     index = int(adjustment * 2 * 12 / 2)
    # else:
    #     index = int((adjustment - 0.5) * 2 * 6 + 6)
    # return index

def loop_and_detect(cam, trt_yolo, conf_th, lib, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    global previous_error
    global start_line
    full_scrn = False
    fps = 0.0
    tic = time.time()
    dt = 0.01
    count = 0
    while True:
        count += 1
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break

        boxes, confs, clss, gpu_util = trt_yolo.detect(img, conf_th)

        if count > 30:
            # Update target_bounding_box
            curr_boxes = len(boxes)
            target_boxes = update_boxex_target(curr_boxes)

            # Update adaptive target processing time
            adaptive_target_processing_time = update_adaptive_target(dt)

            # PID control
            process_time_adjustment = pid_update(dt, adaptive_target_processing_time, dt)

            # Adjustments based on bounding boxes and GPU utilization
            bbox_adjustment = calculate_bbox_adjustment(curr_boxes)
            gpu_utilization_adjustment = calculate_gpu_utilization_adjustment(gpu_util)
            # energy, start_line = energy_per_frame(start_line)
            # print(f"energy per frame {start_line}: {energy}")

            # Combine adjustments and update GPU frequency
            final_adjustment = combine_adjustments(process_time_adjustment, bbox_adjustment, gpu_utilization_adjustment)
            freq_idx = new_freq_index(final_adjustment)
            original_idx = lib.read_frequency()

            #typecasting
            freq_idx = int(freq_idx)
            original_idx = int(original_idx)
            new_freq_idx = ctypes.c_int(freq_idx + original_idx)

            lib.set_freq.argtypes = [ctypes.c_int]
            lib.set_freq(new_freq_idx)  # set the frequency up or down by freq_idx levels

        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        dt = toc - tic
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)



def main():
    lib = ctypes.CDLL('./DFS.so')
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # Init GPU governor
    lib.main()
    lib.set_freq(args.freq)

    # FOR POWER COMSUMP PROFILE
    if args.watts:
        daemon_thread = threading.Thread(target=run_daemon, args=(lib,))
        daemon_thread.start()

    # FOR PERFORMANCE PROFILE
    if args.perf:
        start_time = time.time()

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, lib, vis=vis)

    # FOR POWER COMSUMP PROFILE
    if args.watts:
        lib.stop_daemon()
        daemon_thread.join()

    if args.perf:
        end_time = time.time()
        with open('perf_output.txt', 'a') as file:
            file.write(str(end_time - start_time))
            file.write("\n")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
