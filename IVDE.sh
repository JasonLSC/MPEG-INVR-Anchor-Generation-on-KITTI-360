start_num=$1
num_images=$2

cfg_dir="./IVDE/INVR_cfg/"

./IVDE/IVDE "${cfg_dir}estimation_params.json" "${cfg_dir}SA_sequence_params.json" "${cfg_dir}SA_filenames.json" > ./nerfacto_${start_num}_${num_images}/IVDE_output.txt