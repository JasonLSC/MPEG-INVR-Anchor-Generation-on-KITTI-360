import subprocess
import argparse
import glob
import os


parser = argparse.ArgumentParser()

parser.add_argument('--start_index',type = int,help=' Start index of Data')
parser.add_argument('--num_images',type = int,help='Num images')
config = parser.parse_args()

basedir = f"./kitti360_{config.start_index}_{config.num_images}/"

yuv_name_list = []
# RP1 -> 4
for i in range(1,5):
    yuv_name_list += sorted(glob.glob(os.path.join(basedir, "tmiv_enc", f"QP{i}", "render", "*.yuv")))
# QP0
yuv_name_list += sorted(glob.glob(os.path.join(basedir, "tmiv_enc", f"RP0", "render", "*.yuv")))


for old_name in yuv_name_list:
    
    new_name = old_name.replace("_texture_1408x376_yuv420p10le.yuv", ".png")

    width, height = 1408, 376
    subprocess.run(['ffmpeg', '-s', f'{width}x{height}', '-pix_fmt', 'yuv420p10le', '-i', old_name, '-frames:v', '1', '-c:v', 'png', '-compression_level', '0', new_name], check=True)

