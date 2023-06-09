#!/bin/bash

image_folder="/Users/samiferchiou/VITA/monstereo/data/kitti/training/image_2/"

for ((i=0 ;i<=7480;i++))
do
    image_file="$image_folder"$(printf "%06d.png" "$i")
    python3 -m openpifpaf.predict "$image_file" --json-output /Users/samiferchiou/VITA/monstereo/data/kitti-pifpaf/annotations_car_2 --checkpoint=shufflenetv2k16-apollo-24 --instance-threshold 0.05
done
