import glob
import json
import shutil
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--start_index',type = int,help=' Start index of Data')
parser.add_argument('--num_images',type = int,help='Num images')
parser.add_argument('--QP_level',type = str,
                    choices=["RP0", "QP1", "QP2", "QP3", "QP4"], help='QP')

config = parser.parse_args()

### rename decoded YUV
filename_list = \
    sorted(glob.glob(f"./kitti360_{config.start_index}_{config.num_images}/tmiv_enc/{config.QP_level}/TMIV_G1_Z_{config.QP_level}_tex_*_1408x376_yuv420p10le.yuv"))

with open("./TMIV/config/ctc/sequences/Z.json", "r") as file:
    data = json.load(file)

source_cam_name = data["sourceCameraNames"]
dirname = os.path.dirname(filename_list[0])

new_filename_list = []
for idx, old_filename in enumerate(filename_list):
    new_filename = os.path.join(dirname, \
                                source_cam_name[idx] + "_texture_1408x376_yuv420p10le.yuv")
    shutil.copy(old_filename, new_filename)

    new_filename_list.append(new_filename)

### generate "SA_filename.json" as cfg file for IVDE
IVDE_filename_dict = {
    "filenames": []
}

for new_filename in new_filename_list:
    new_depth_filename = new_filename.replace("texture", "depth").replace("yuv420p10le", "yuv420p16le")

    IVDE_filename_dict["filenames"].append(
        {"InputView": new_filename,
         "OutputDepthMap": new_depth_filename}
    )

with open(os.path.join(dirname, "IVDE_filenames.json"), 'w') as file:
    json.dump(IVDE_filename_dict, file, indent=4)