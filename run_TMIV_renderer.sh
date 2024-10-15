start_num=$1
num_images=$2

# QP1-4
# for i in $(seq $((start_num+1)) 2 $((start_num+num_images-1)))
# do

#     for j in 1 2 3 4
#     do
#         echo "Rendering QP$j view ${i}_00"
#         ./TMIV/out/install/gcc-release/bin/TmivRenderer -n 1 -N 1 -s Z -f 0 -r QP$j -v ${i}_00 \
#             -c ./TMIV/config/ctc/invr_dsde_anchor/G_6_TMIV_render.json \
#             -p configDirectory ./TMIV/config \
#             -p inputDirectory  ./kitti360_${start_num}_${num_images}/tmiv_enc  \
#             -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc &

#         ./TMIV/out/install/gcc-release/bin/TmivRenderer -n 1 -N 1 -s Z -f 0 -r QP$j -v ${i}_01 \
#             -c ./TMIV/config/ctc/invr_dsde_anchor/G_6_TMIV_render.json \
#             -p configDirectory ./TMIV/config \
#             -p inputDirectory  ./kitti360_${start_num}_${num_images}/tmiv_enc \
#             -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc
#     done

# done

# RP0
for i in $(seq $((start_num+1)) 2 $((start_num+num_images-1)))
do
    ./TMIV/out/install/gcc-release/bin/TmivRenderer -n 1 -N 1 -s Z -f 0 -r RP0 -v ${i}_00 \
        -c ./TMIV/config/ctc/invr_dsde_anchor/G_6_TMIV_render.json \
        -p configDirectory ./TMIV/config \
        -p inputDirectory  ./kitti360_${start_num}_${num_images}/tmiv_enc \
        -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc

    ./TMIV/out/install/gcc-release/bin/TmivRenderer -n 1 -N 1 -s Z -f 0 -r RP0 -v ${i}_01 \
        -c ./TMIV/config/ctc/invr_dsde_anchor/G_6_TMIV_render.json \
        -p configDirectory ./TMIV/config \
        -p inputDirectory  ./kitti360_${start_num}_${num_images}/tmiv_enc \
        -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc

done