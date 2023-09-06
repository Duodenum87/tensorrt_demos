#!/bin/bash
arr=(921600000 844800000 768000000 691200000 614400000 537600000 460800000 384000000 307200000 230400000 153600000 76800000)
    echo "921600000" | tee /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq
    for x in "${arr[@]}"
    do
        echo $x | tee /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq

        python trt_yolo.py -m yolov4-tiny-416 --video video/cat.mp4 -p 100000000 -hf $x -lf $x     
    done
