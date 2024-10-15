#!/bin/bash
start_num=700
num_images=5

bash kitti2omaf.sh $start_num $num_images
bash IVDE.sh $start_num $num_images
bash depth_range.sh $start_num $num_images
bash render.sh $start_num $num_images