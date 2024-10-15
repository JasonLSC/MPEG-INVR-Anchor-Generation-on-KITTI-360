import json
import matplotlib.pyplot as plt
import numpy as np

rd_file = "./nerfacto_700_5/rd/rd.json"

with open(rd_file, "r") as file:
    rd_dict = json.load(file)

rate = []
psnr = []
ssim = []
lpips = []

for QP in ["QP1", "QP2", "QP3", "QP4"]:
    rate.append(rd_dict[QP]["bitrate"])
    psnr.append(rd_dict[QP]["PSNR"])
    ssim.append(rd_dict[QP]["SSIM"])
    lpips.append(rd_dict[QP]["LPIPS_A"])

x = np.array(rate)
y1 = np.array(psnr)
y2 = np.array(ssim)
y3 = np.array(lpips)

# 创建一个包含三个子图的图形，排列为一行三列
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制第一个子图
axs[0].plot(x, y1, 'b', label="QP1-4")
axs[0].set_title('PSNR - Bitrate')
axs[0].set_xlabel('Bitrate (Mb/s)')
axs[0].set_ylabel('PSNR (dB)')
axs[0].axhline(y=rd_dict["RP0"]["PSNR"], color='black', linestyle='--', label="RP0")
axs[0].legend()

# 绘制第二个子图
axs[1].plot(x, y2, 'b', label="QP1-4")
axs[1].set_title('SSIM - Bitrate')
axs[1].set_xlabel('Bitrate (Mb/s)')
axs[1].set_ylabel('SSIM')
axs[1].axhline(y=rd_dict["RP0"]["SSIM"], color='black', linestyle='--', label="RP0")
axs[1].legend()

# 绘制第三个子图
axs[2].plot(x, y3, 'b', label="QP1-4")
axs[2].set_title('LPIPS - Bitrate')
axs[2].set_xlabel('Bitrate (Mb/s)')
axs[2].set_ylabel('LPIPS')
axs[2].axhline(y=rd_dict["RP0"]["LPIPS_A"], color='black', linestyle='--', label="RP0")
axs[2].legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()