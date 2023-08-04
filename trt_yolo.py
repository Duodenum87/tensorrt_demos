"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import signal
import subprocess
import threading
from tqdm import tqdm

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'
keep_running = True
# frames = 0

# def print_frame(signum, frame):
#     print(frames)

# def reset_frame(signum, frame):
#     frames = 0

# signal.signal(signal.SIGUSR1, reset_frame)
# signal.signal(signal.SIGUSR2, print_frame)

def read_file(path):
    with open(path, "r") as f:
        power = f.read().strip()
    return power

def get_power_consump():
    global keep_running
    sleep_time = 0.1
    energy = [0, 0, 0]
    power_path = ["/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input",
                    "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input",
                    "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input"]

    last_time = time.time()
    while keep_running:
        curr_time = time.time()
        dt = curr_time - last_time
        for i in range(3):
            power = read_file(power_path[i])
            energy[i] += float(power) * dt
        last_time = curr_time
        time.sleep(sleep_time)
        
    with open('gpu_output.txt', 'a') as file:
        for i in range(3):
            file.write(f"{energy[i]}\n")
    

def get_gpu_usage():
    global keep_running
    target_time_per_loop = 0.1  # target time per loop in seconds
    while keep_running:
        start_time = time.time()

        process = subprocess.Popen(['cat', '/sys/devices/platform/host1x/57000000.gpu/load'], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        gpu_load = output.decode('utf-8').strip()
        with open('gpu_output.txt', 'a') as file:
            file.write(str(gpu_load) + '\n')

        end_time = time.time()
        elapsed_time = end_time - start_time

        sleep_time = target_time_per_loop - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
    
def GPU_scale_down(GPU_freq_stats):
    GPU_freq = [76800000, 153600000, 230400000, 307200000, 384000000, 460800000, 
                537600000, 614400000, 691200000, 768000000, 844800000, 921600000]
    if GPU_freq_stats != 0: 
        freq = GPU_freq[GPU_freq_stats - 1] 
    else:
        freq = GPU_freq[0]

    command = "echo " + str(freq) + "| tee /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq"
    subprocess.run(command, shell=True)
    return GPU_freq_stats - 1 if GPU_freq_stats != 0 else 0
    
def GPU_scale_up(GPU_freq_stats):
    GPU_freq = [76800000, 153600000, 230400000, 307200000, 384000000, 460800000, 
                537600000, 614400000, 691200000, 768000000, 844800000, 921600000]
    if GPU_freq_stats != len(GPU_freq) - 1: 
        freq = GPU_freq[GPU_freq_stats + 1] 
    else:
        freq = GPU_freq[len(GPU_freq) - 1]

    command = "echo " + str(freq) + "| tee /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq"
    subprocess.run(command, shell=True)
    return GPU_freq_stats + 1 if GPU_freq_stats != len(GPU_freq) - 1 else len(GPU_freq) - 1

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
    parser.add_argument("-p" ,"--period", type = int, help="how many frames into next frequency?")
    parser.add_argument("-lf", "--lowFreq")
    parser.add_argument("-hf", "--highFreq")
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, progress_bar, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    count = 0
    """
    """
    GPU_freq_stats = 11
    last_rate = None
    max_rate = 0
    while True:
        freqflag = 0
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        # For the scaling overhead test
        if count == period:
            freqflag = 2
        elif count == 2 * period:
            freqflag = 1
            count = 0
        ####

        boxes, confs, clss = trt_yolo.detect(img, highfreq, lowfreq, freqflag, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)

        """ Scale by the estimated remaining time
        """
        progress_bar.update()
        curr_rate = progress_bar.format_dict['rate']
        if curr_rate > max_rate:
            max_rate = curr_rate
        if count % 10 == 0:
            # if first_rate is None:
            #     fisrt_rate = curr_rate
            # else:
            #     if abs(first_rate - curr_rate) < 0.01 * curr_rate:
            #         GPU_freq_stats = GPU_scale_down(GPU_freq_stats)
            #     elif curr_rate < 0.8 * first_rate:
            #         GPU_freq_stats = GPU_scale_up(GPU_freq_stats)
            if curr_rate < 0.9 * max_rate:
                GPU_freq_stats = GPU_scale_up(GPU_freq_stats)
            elif last_rate is not None and abs(last_rate - curr_rate) < 0.1 * curr_rate:
                GPU_freq_stats = GPU_scale_down(GPU_freq_stats)
        last_rate = curr_rate

        """
        """
        # frames = frames + 1
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    # CODE BLOCK for cutting frames
        # if count == 100:
        #     with open('fps_output.txt', 'a') as file:
        #         file.write(str('--------------') + '\n')
        #     with open('gpu_output.txt', 'a') as file:
        #         file.write(str('--------------') + '\n')
        # if count == 250:
        #     with open('fps_output.txt', 'a') as file:
        #         file.write(str('--------------') + '\n')
        #     with open('gpu_output.txt', 'a') as file:
        #         file.write(str('--------------') + '\n')
        # with open('fps_output.txt', 'a') as file:
            # file.write(str(fps) + '\n')
            
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        count=count+1
        print(count)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    global keep_running
    with open('gpu_output.txt', 'a') as file:
        file.write("\n")
    t = threading.Thread(target=get_power_consump)
    t.start()

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    global period
    period = args.period
    global highfreq
    highfreq = args.highFreq
    global lowfreq
    lowfreq = args.lowFreq
    # Create a progress bar
    progress_bar = tqdm(total=cam.get_frames())

    loop_and_detect(cam, trt_yolo, args.conf_thresh, progress_bar, vis=vis)

    keep_running = False
    progress_bar.close()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    with open('fps_output.txt', 'a') as file:
        file.write(str(end_time - start_time))
        file.write("\n")
