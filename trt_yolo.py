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
    args = parser.parse_args()
    return args

def run_daemon(lib):
    lib.power_monitoring_daemon()

# PID Controller Variables
Kp = 10 # Proportional gain
Ki = 0.001  # Integral gain
Kd = 0.05  # Derivative gain

integral_error = 0
previous_error = 0

recent_perfs = []
perf_entries = 10

# Adaptively choose performance target
def update_adaptive_target(current_processing_time):
    recent_perfs.append(current_processing_time)
    if len(recent_perfs) > perf_entries:
        recent_perfs.pop(0)
    adaptive_target = sum(recent_perfs) / len(recent_perfs)
    return adaptive_target

def normalize_output(pid_output, min_output, max_output):
    return (pid_output - min_output) / (max_output - min_output)

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

    # Debugging print statements
    print(f"Total Output: {total_output}")

    return total_output

def calculate_bbox_adjustment(current_bounding_boxes, target_bounding_boxes):
    deviation = current_bounding_boxes - target_bounding_boxes
    scaling_factor = ...  # Define scaling factor
    return deviation / target_bounding_boxes * scaling_factor

def calculate_gpu_utilization_adjustment(current_gpu_utilization, target_gpu_utilization):
    lower_bound, upper_bound = target_gpu_utilization
    if current_gpu_utilization < lower_bound:
        return (current_gpu_utilization - lower_bound) / lower_bound
    elif current_gpu_utilization > upper_bound:
        return (current_gpu_utilization - upper_bound) / (100 - upper_bound)
    return 0

# Function to map the adjustment value to a GPU frequency
def new_freq_index(adjustment):
    index = min(max(int(adjustment * 12), 0), 12 - 1)# len(freqs) = 12
    return index
    # if adjustment < 0.5:
    #     index = int(adjustment * 2 * 12 / 2)
    # else:
    #     index = int((adjustment - 0.5) * 2 * 6 + 6)
    # return index

def combine_adjustments(perf_adj, bbox_adj, gpu_adj):
    weight_perf = 1
    weight_bbox = 0.2
    weight_gpu = 0.2
    return weight_perf * perf_adj + weight_bbox * bbox_adj + weight_gpu * gpu_adj

def loop_and_detect(cam, trt_yolo, conf_th, lib, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    global previous_error
    full_scrn = False
    fps = 0.0
    last_time = time.time()
    count = 0
    exceed_flag = 0

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break

        current_time = time.time()
        dt = current_time - last_time

        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        #
        curr_boxes = len(boxes)
        curr_gpu_util = lib.read_GPU_usage()

         # Update adaptive target processing time
        adaptive_target_processing_time = update_adaptive_target(current_time - last_time)

         # PID control
        pid_output = pid_update(current_time - last_time, adaptive_target_processing_time, dt)
        process_time_adjustment = normalize_output(pid_output, -2, 1)

        # Adjustments based on bounding boxes and GPU utilization
        # bbox_adjustment = calculate_bbox_adjustment(curr_boxes, target_bounding_boxes)
        # gpu_utilization_adjustment = calculate_gpu_utilization_adjustment(curr_gpu_util, target_gpu_utilization)
        bbox_adjustment = 0
        gpu_utilization_adjustment = 0

        # Combine adjustments and update GPU frequency
        final_adjustment = combine_adjustments(process_time_adjustment, bbox_adjustment, gpu_utilization_adjustment)
        freq_idx = new_freq_index(final_adjustment)
        lib.set_freq.argtypes = [ctypes.c_int]
        lib.set_freq(freq_idx)  # Implement GPU frequency setting logic

        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - last_time)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        count += 1
        # with open('perf_output.txt', 'a') as file:
        #     file.write(str(count))
        #     file.write(' ')
        #     file.write(str(time.time() - start_time))
            # file.write("\n")
        last_time = current_time



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

    # global keep_running
    # with open('gpu_output.txt', 'a') as file:
    #     file.write("\n")
    # t = threading.Thread(target=get_power)
    # t.start()
    # daemon_thread = threading.Thread(target=run_daemon, args=(lib,))
    # daemon_thread.start()
    usage = lib.read_GPU_usage()

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    usage = lib.read_GPU_usage()
    with open('perf_output.txt', 'a') as file:
        file.write(str(time.time() - start_time))
        file.write("\n")

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    # progress_bar = tqdm(total=cam.get_frames())
    loop_and_detect(cam, trt_yolo, args.conf_thresh, lib, vis=vis)

    # lib.stop_daemon()
    # with open('power_consump.txt', 'a') as file:
    #     file.write("stop")
    #     file.write("\n")
    # daemon_thread.join()
    # keep_running = False
    # progress_bar.close()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    with open('perf_output.txt', 'a') as file:
        file.write(str(end_time - start_time))
        file.write("\n")

