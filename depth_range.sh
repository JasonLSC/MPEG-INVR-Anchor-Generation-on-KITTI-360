start_num=$1
num_images=$2

filename="nerfacto_${start_num}_${num_images}/IVDE_output.txt"

python rewrite_depth_range.py \
    --start_index $start_num --num_images $num_images \
     --drop_out
    #  --z_near $first_number --z_far $second_number