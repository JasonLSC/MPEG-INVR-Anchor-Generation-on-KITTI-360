start_num=$1
num_images=$2

# QP_indices=(1 2 3 4)

# QP1-4
for i in "${QP_indices[@]}"
do
    ./TMIV/out/install/gcc-release/bin/TmivDecoder -n 1 -N 1 -s Z -r QP$i \
    -c ./TMIV/config/ctc/invr_dsde_anchor/G_4_TMIV_decode.json \
    -p configDirectory ./TMIV/config \
    -p inputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc \
    -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc
done 

# RP0
./TMIV/out/install/gcc-release/bin/TmivDecoder -n 1 -N 1 -s Z -r RP0 \
    -c ./TMIV/config/ctc/invr_dsde_anchor/G_4_TMIV_decode.json \
    -p configDirectory ./TMIV/config \
    -p inputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc \
    -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc
