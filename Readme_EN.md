# Overview
The current experiment in MPEG-INVR anchor generation on KITTI-360 is conducted on frames 700-704 from the 2013_05_28_drive_0000_sync folder.

You can simply run `run_dsde.sh` to execute the process.

# Dataset preparation
For KITTI360 Anchor generation, the starting point of the entire project is to obtain scene configuration files and YUV image files that conform to the MPEG OMAF format.

We have provided the relevant files in the kitti360_700_5 folder. The folder includes: 
1. Stereo RGB images stored in PNG format from KITTI-360, such as 700_00.png and 700_01.png, where xxx_00 and xxx_01 form a pair of stereo images; 
2. A scene description file that conforms to the MPEG OMAF format: mpeg_omaf.json; 
3. A configuration file that conforms to the NeRF format: transforms.json; 
4. YUV files converted from the stereo images in "1."

# Configure files
To ensure the smooth progress of the upcoming experiments, here are our configuration files：
- Sequences cfg file named `Z.json` stored in `./miv_cfg/sequences/Z.json`
  - (Compared to the MPEG OMAF file mentioned above, the only differences are in the naming of the source view and the sourceCameraNames. These modifications were made to facilitate running MIV experiments.)
- MIV DSDE mode cfg files stored in `./miv_cfg/invr_dsde_anchor`

# Software
- TMIV 20.0.0
- IVDE

# Run TMIV DSDE
I have attached the scripts used during my MIV DSDE mode experiment:
- `run_dsde.sh`
  - `run_TMIV_enc.sh`
  - `run_VVC_enc.sh`
  - `run_VVC_dec.sh`
  - `run_TMIV_dec.sh`
  - `rename_decoded_YUV.sh`
  - `run_IVDE.sh`
  - `run_TMIV_renderer.sh`
  - `run_yuv2png.sh`
## TMIV Enc.
`run_TMIV_enc.sh` is the script I used to run the TMIV encoder to generate the atlas.
## VVC Enc.
`run_VVC_enc.sh` is the script I used to run the VVC encoder(VVenC, actually) to compress atlas. This script will concurrently execute encoding experiments at the four bitrate points QP1-4.
## VVC Dec.
`run_VVC_dec.sh` is the script I used to run the VVC decoder(VVdeC, actually) to decompress the bistreams of atlases.
## TMIV Dec.
`run_TMIV_dec.sh` is the script I used to run the TMIV decoder to get reconstructed input views.
## Scripts for renaming
`rename_decoded_YUV.sh` is the script I used to rename reconstructed input views for following operations.
## IVDE
`run_IVDE.sh` is the script I used to run IVDE to obatin decoder side estimated depth maps.
## TMIV Rendering
`run_TMIV_renderer.sh` is the script I used to render test views from reconstructed input views and corresponding decoder side estimated depth maps. (RP0-4)
## YUV to PNG
`run_yuv2png.sh` is the script I used to convert rendered images from YUV format to PNG format for following metric calculation.

# Misc.
I also share the jupyter notebook `get_excel_tab.ipynb`, which I used to calculate the quality metrics of the renderings and the total bitrate.

# Contact information
If you have any questions about this anchor generation project, feel free to reach out to Sicheng Li at jasonlisicheng@zju.edu.cn.