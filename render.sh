start_num=$1
num_images=$2



for i in $(seq $((start_num+1)) 2 $((start_num+num_images-1)))
do
    ./TMIV/out/install/gcc-release/bin/TmivRenderer -n 1 -N 1 -s Z -f 0 -r R0 -v ${i}_00 \
        -c ./TMIV/config/ctc/best_reference/R_1_TMIV_render_INVR.json \
        -p configDirectory ./TMIV/config \
        -p inputDirectory  ./nerfacto_${start_num}_${num_images} \
        -p outputDirectory ./nerfacto_${start_num}_${num_images}/RP0/render

    ./TMIV/out/install/gcc-release/bin/TmivRenderer -n 1 -N 1 -s Z -f 0 -r R0 -v ${i}_01 \
        -c ./TMIV/config/ctc/best_reference/R_1_TMIV_render_INVR.json \
        -p configDirectory ./TMIV/config \
        -p inputDirectory  ./nerfacto_${start_num}_${num_images} \
        -p outputDirectory ./nerfacto_${start_num}_${num_images}/RP0/render

done