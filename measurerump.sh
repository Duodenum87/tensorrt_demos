#!/bin/bash
# rails=("VDD_Total /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"
# "VDD_GPU /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input"
# "VDD_CPU /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input")
arr=(76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000)
arr2=(921600000 844800000 768000000 691200000 614400000 537600000 460800000 384000000 307200000 230400000 153600000 76800000)
# arr2=(921600000)
for x in "${arr[@]}"
do
    for y in "${arr2[@]}"
    do
        # echo "$x" "$y" | tee -a result.txt resultPower.txt
        # period=(1 5 10 100 500 1000 1250 2000 2500 5000)
        period=(1 2	3 4	5 10 20 50 100 250)
        for z in "${period[@]}"
        do
            for i in {1}
            do
                
                echo $y | tee /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq
                # Performance TEST
                cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/trans_stat | tee -a freq.txt
                { time python trt_yolo.py -m yolov4-tiny-416 --video video/dog.mp4 -p $z -hf $y -lf $x; } 2>&1 >/dev/null | grep real | awk '{print $2}' >> result.txt     

                cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/trans_stat | tee -a freq.txt


                # Power TEST
                # for ((i = 0; i < ${#rails[@]}; i++));
                # do
                #     read -r name[$i] node[$i] pwr_area[$i] int_count[$i]<<<"$(echo "${rails[$i]} 0 0")"
                # done 

                # PID=$(ps -f -U root -u root | grep "python3 trt_googlenet.py" | awk 'NR==1 {print $2}')

                # while kill -0 $PID &>/dev/null;
                # do
                #     star=$(date +%s%3N)
                #     for ((i = 0; i < ${#rails[@]}; i++));
                #     do
                #         # mW * ms = 0.000001 * W * s = 0.000001 J
                #         temp=$(cat "${node[$i]}")
                #         temp2=$(( temp*gap ))
                #         pwr_area[$i]=$(( ${pwr_area[$i]}+temp2 ))
                #         int_count[$i]=$((${int_count[$i]}+1))
                #     done
                #     end=$(date +%s%3N)
                #     gap=$(($end-$star))
                # done
                # echo "${pwr_area[0]}  ${pwr_area[1]}   ${pwr_area[2]}   ${int_count[0]}" | tee -a resultPower.txt
            done
        done
    done
    unset arr2[${#arr2[@]}-1]
done
