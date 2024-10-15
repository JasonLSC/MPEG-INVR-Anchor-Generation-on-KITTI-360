start_num=$1
num_images=$2

# RP1 -> RP4: QP: 24 30 36 42
QPs=(24 30 36 42)
length=${#QPs[@]}

atlas_ids=(0 1 2 3)

for ((i=0; i<length; i++))
do
    for j in "${atlas_ids[@]}"
    do 
        ./TMIV/out/install/gcc-release/bin/vvdecapp \
        -b ./kitti360_${start_num}_${num_images}/tmiv_enc/QP$((i+1))/TMIV_G1_Z_QP$((i+1))_tex_c0$((j)).bit \
        -o ./kitti360_${start_num}_${num_images}/tmiv_enc/QP$((i+1))/TMIV_G1_Z_QP$((i+1))_tex_c0$((j))_1408x6320_yuv420p10le.yuv &
    done
done

wait

echo "VVC decoding tasks completed."
