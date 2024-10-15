start_num=$1
num_images=$2

python kitti2json.py --mode nerfacto --start_index $start_num --num_images $num_images
python camorph/format_convert.py --start_index $start_num --num_images $num_images --drop_out 