start_num=$1
num_images=$2

cfg_dir="./IVDE/INVR_cfg/"


# for i in 1 2 3 4
# do
#     YUV_basedir="./kitti360_${start_num}_${num_images}/tmiv_enc/QP$i"

#     ./IVDE/IVDE "${cfg_dir}estimation_params.json" "${cfg_dir}SA_sequence_params.json" "${YUV_basedir}/IVDE_filenames.json" \
#         > ${YUV_basedir}/IVDE_output.txt
# done

YUV_basedir="./kitti360_${start_num}_${num_images}/tmiv_enc/RP0"

./IVDE/IVDE "${cfg_dir}estimation_params.json" "${cfg_dir}SA_sequence_params.json" "${YUV_basedir}/IVDE_filenames.json" \
    > ${YUV_basedir}/IVDE_output.txt