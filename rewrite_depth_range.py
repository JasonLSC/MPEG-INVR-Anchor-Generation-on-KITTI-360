import json
import argparse
import re



parser = argparse.ArgumentParser()

parser.add_argument('--start_index',type = int,help=' Start index of Data')
parser.add_argument('--num_images',type = int,help='Num images')
parser.add_argument('--z_near',type = float,help='z_near')
parser.add_argument('--z_far',type = float,help='z_far')
parser.add_argument('--drop_out', action="store_true")

config = parser.parse_args()


''' Read Z_near and Z_far from log of IVDE '''
def read_numbers_from_line(filename, line_number):
    with open(filename, 'r') as file:
        # 读取所有行
        lines = file.readlines()
        
        # 检查行号是否超出范围
        if line_number > len(lines):
            raise ValueError(f"File has only {len(lines)} lines, but line {line_number} was requested.")
        
        # 获取指定行
        line_content = lines[line_number - 1].strip()

        match = re.search(r'\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]', line_content)
        
        if match:
            first_number = float(match.group(1))
            second_number = float(match.group(2))
            return first_number, second_number
        else:
            raise ValueError("No valid number pair found in the string.")


z_near, z_far = read_numbers_from_line(f"nerfacto_{config.start_index}_{config.num_images}/IVDE_output.txt", 17)



basedir = f'/work/Users/lisicheng/Code/INVR-KITTI360/nerfacto_{config.start_index}_{config.num_images}/'

with open(basedir+"mpeg_omaf.json", 'r') as jsonfile:
    data = json.load(jsonfile)

data["Version"] = "4.0"
data["sourceCameraNames"] = []
#"BoundingBox_center": [0, 0, 0],
data["BoundingBox_center"] = [0, 0, 0]
data["lengthsInMeters"] = True

for idx, item in enumerate(data["cameras"]):
    data["cameras"][idx]["Depth_range"] = [ z_near, z_far ]
    data["cameras"][idx]["Name"] = data["cameras"][idx]["Name"].replace("_texture_1408x376_yuv420p10le.yuv", "")

    if config.drop_out:
        if idx % 4 == 0 or idx % 4 == 1:
            data["sourceCameraNames"].append(data["cameras"][idx]["Name"])

save_dir = "/work/Users/lisicheng/Code/INVR-KITTI360/TMIV/config/ctc/sequences/"

with open(save_dir + 'Z.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)