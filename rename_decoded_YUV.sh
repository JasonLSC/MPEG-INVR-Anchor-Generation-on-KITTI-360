start_num=$1
num_images=$2


for i in 1 2 3 4
do
    python rename_decoded_YUV.py \
        --start_index $start_num --num_images $num_images \
        --QP_level QP$i 
done

python rename_decoded_YUV.py \
    --start_index $start_num --num_images $num_images \
    --QP_level RP0