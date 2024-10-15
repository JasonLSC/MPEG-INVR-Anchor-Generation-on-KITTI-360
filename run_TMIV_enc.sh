start_num=$1
num_images=$2

./TMIV/out/install/gcc-release/bin/TmivEncoder -n 1 -s Z -f 0 \
    -c ./TMIV/config/ctc/invr_dsde_anchor/G_1_TMIV_encode.json \
    -p configDirectory ./TMIV/config \
    -p inputDirectory  ./kitti360_${start_num}_${num_images} \
    -p outputDirectory ./kitti360_${start_num}_${num_images}/tmiv_enc
