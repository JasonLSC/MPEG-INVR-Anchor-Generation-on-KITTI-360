start_num=$1
num_images=$2

# RP1 -> RP4: QP: 24 30 36 42
QPs=(24 30 36 42)
length=${#QPs[@]}

for ((i=0; i<length; i++))
do
    mkdir ./kitti360_${start_num}_${num_images}/tmiv_enc/QP$((i+1))
done

atlas_ids=(0 1 2 3)

for ((i=0; i<length; i++))
do
    for j in "${atlas_ids[@]}"
    do 
        ./TMIV/out/install/gcc-release/bin/vvencFFapp \
            -c ./TMIV/config/ctc/invr_dsde_anchor/G_2_VVenC_encode_tex.cfg \
        -i ./kitti360_${start_num}_${num_images}/tmiv_enc/RP0/TMIV_G1_Z_RP0_tex_c0$((j))_1408x6320_yuv420p10le.yuv \
        -b ./kitti360_${start_num}_${num_images}/tmiv_enc/QP$((i+1))/TMIV_G1_Z_QP$((i+1))_tex_c0$((j)).bit \
        -s 1408x6320 -q ${QPs[$i]} -f 1 -fr 30 &
    done
done

wait

echo "All tasks completed."
