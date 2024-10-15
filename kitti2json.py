import os,imageio
# import torch
import numpy as np
import json
import argparse

'''
Output: images, poses, bds, render_pose, itest;
poses (c2w)
'''

def main(datadir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type = str,help='Datasets type file')
    parser.add_argument('--start_index',type = int,help=' Start index of Data')
    parser.add_argument('--num_images',type = int,help='Num images')
    parser.add_argument('--use_fisheye',action='store_true')
    parser.add_argument('--f2nerf',action='store_true')
    parser.add_argument('--use_semantic',action='store_true')
    config = parser.parse_args()


    poses, imgs, K_00,all_name =_load_data(datadir,
                                            start_index = config.start_index,
                                            num = config.num_images)
    if config.use_fisheye:
        fisheye_poses, fisheye_imgs, K_02, K_03,all_name_fisheye = load_fisheye(datadir,
                                                                        start_index = config.start_index,
                                                                        num = config.num_images)
    ## 对加上fisheye 的相机进行 Normalize    
    if config.use_fisheye:
        bbx2world = poses[0].copy()
        ## fisheye 的 Pose 需要和 perspective camera pose 统一 Normalizie
        poses = Normailize_T_fisheye(poses,config.mode,fisheye_poses=fisheye_poses)
        print("Use fisheye!")
    else:
        bbx2world = poses[0].copy()
        ## 设第一张相机的Pose 是单位矩阵，对其他相机的Pose 需要进行调整为相对于第一帧的Pose 相对位姿
        poses,inv_pose,scale = Normailize_T(poses,config.mode)   ## 对于 平移translation 进行归一化

    
    """whether include semantic image"""
    semantic_imgs = None
    if config.use_semantic:
        semantic_imgs = _load_semantic_imgs(datadir=datadir,sequence='2013_05_28_drive_0000_sync',start_index = config.start_index, num = config.num_images)
    
    if config.f2nerf:
        focal = K_00[0][0]
        F2Nerf(poses,focal,imgs,K_00)
        # F2Nerf_fisheye(poses,fisheye_imgs,K_02,K_03)
        print(" End!")
        exit()

    
    if config.mode == 'nerfacto':
        nerfacto_tojson(poses,imgs,K_00,bbx=bbx2world,image_names=all_name,config = config,inv_pose=inv_pose,scale=scale,semantic_imgs = semantic_imgs)
    elif config.mode == 'neus':
        neus_tojson(poses,imgs,K_00)
    else:
        raise ValueError("Specify which mode nerfacto or neus")

    if config.mode == 'nerfacto' and config.use_fisheye:
        nerfacto_fisheye_tojson(poses,fisheye_imgs,K_02,K_03,all_name_fisheye)
    

    return 

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    # x = normalize(np.cross(z, y_))  # (3)
    x = normalize(np.cross(y_, z))

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    # y = np.cross(x, z)  # (3)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def neus_tojson(poses,imgs,K):
    ## 创建一个 关于scene box 的字典
    scene_min, scene_max = -6.0, 6.0
    scene_box = {
        "aabb":listify_matrix(np.array([[scene_min,scene_min,scene_min],[scene_max,scene_max,scene_max]])),
        "near": 0,
        "far": 6,
        "radius":1,
         "collider_type": "near_far"
    }

    out_data = {
        'cameara_model': "OPENCV",
        'width': imgs[0].shape[1],
        'height': imgs[0].shape[0],
        "has_mono_prior": False,
        "scene_box": scene_box
        }

    out_data['frames'] = []

    instrintic = np.eye(4)
    instrintic[:3,:3] = K

    os.makedirs("kitti360_neus",exist_ok=True)
    for i in range(0,poses.shape[0]):
        frame_data = {
            'rgb_path': "./{:02d}".format(i)+".png",
            'transform_matrix': listify_matrix(poses[i]),
            'intrinsics': listify_matrix(instrintic),
            'mono_depth_path': "{:02d}_depth.npy".format(i),
            'mono_normal_path': "{:02d}_normal.npy".format(i),
        }
        filename ='kitti360_neus/'+ "{:02d}".format(i)+".png"
        imageio.imwrite(filename,imgs[i])

        out_data['frames'].append(frame_data)

    with open(f'kitti360_neus/meta_data.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

    return

def F2Nerf(poses,focal,imgs,K):   
    num_images = imgs.shape[0]
    i_all = np.arange(num_images)
    
    i_train = [ i for i in i_all if i % 4 == 0 or i %4 == 1]
    i_eval = [i for i in i_all if i % 4 == 2]
    
    train_pose = poses[i_train][:,:3,:]
    train_image = imgs[i_train]
    test_pose = poses[i_eval][:,:3,:]
    distortion = np.zeros((4,1))
    bound =np.array([0,6])

    meat_list = []
    for i in np.arange(len(i_train)):
        pose = train_pose[i].reshape(-1,1)
        K = K.reshape(-1,1)
        meat_list.append(np.concatenate([pose,K,distortion,bound[:,None]],axis=0))
    cams_meta = np.array(meat_list).squeeze()
    base_dir = "f2nerf_data/images/" 
    os.makedirs(base_dir,exist_ok=True)
    np.save(base_dir + "cams_meta.npy",cams_meta)
    for i in range(train_image.shape[0]):
        filename = base_dir + "{:02d}".format(i) +".png"
        imageio.imwrite(filename,train_image[i])

    ## test_poses
    np.save(base_dir + "poses_render.npy",test_pose)
    return


def _load_semantic_imgs(datadir,sequence = None,start_index = 0, num = 0, seg_sky = False):
    semantic_imgs = [] 
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)

    root_dir = os.path.join(os.path.join(datadir, 'data_2d_semantics/train'), sequence)
    semantic_img_00 = os.path.join(root_dir,os.path.join("image_00","semantic")) 
    semantic_img_01 = semantic_img_00.replace("image_00","image_01")
    for idx in range(start_index,start_index+num,1):    
        img_00 = imread(os.path.join(semantic_img_00,"{:010d}.png").format(idx))
        img_01 = imread(os.path.join(semantic_img_01,"{:010d}.png").format(idx))
        if seg_sky:
            img_00[img_00 != 23 ] = 255
            img_00[img_00 == 23 ] = 0
            img_01[img_01 != 23 ] = 255
            img_01[img_01 == 23 ] = 0

        semantic_imgs.append(img_00)
        semantic_imgs.append(img_01)

    return semantic_imgs


def nerfacto_fisheye_tojson(poses,fisheye_imgs,K_02,K_03,all_name_fisheye):
    numimgs = fisheye_imgs.shape[0]
    poses_fisheye = poses[-numimgs:]

    out_data = {
        'k1_02': K_02['distortion_parameters']['k1'],
        'k2_02': K_02['distortion_parameters']['k2'],
        'gamma1_02':K_02['projection_parameters']['gamma1'],
        'gamma2_02':K_02['projection_parameters']['gamma2'],
        'u0_02':K_02['projection_parameters']['u0'],
        'v0_02':K_02['projection_parameters']['v0'],
        "mirror_02": K_02['mirror_parameters']['xi'],

        'k1_03': K_03['distortion_parameters']['k1'],
        'k2_03': K_03['distortion_parameters']['k2'],
        'gamma1_03':K_03['projection_parameters']['gamma1'],
        'gamma2_03':K_03['projection_parameters']['gamma2'],
        'u0_03':K_03['projection_parameters']['u0'],
        'v0_03':K_03['projection_parameters']['v0'],
        "mirror_03": K_03['mirror_parameters']['xi'],

        'w': fisheye_imgs[0].shape[1],
        'h': fisheye_imgs[0].shape[0]
        }
    out_data['frames'] = []

    for i in range(0,poses_fisheye.shape[0]):
        frame_data = {
            'file_path': all_name_fisheye[i]+".png",
            'transform_matrix': listify_matrix(poses_fisheye[i])
        }
        filename ='kitti360_nerfacto/'+ all_name_fisheye[i] +".png"
        imageio.imwrite(filename,fisheye_imgs[i])

        out_data['frames'].append(frame_data)

    with open(f'kitti360_nerfacto/transforms_fisheye.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)


    return

def nerfacto_tojson(poses,imgs,K,bbx=None,image_names=None,config=None,inv_pose=None,scale=1,semantic_imgs=None):
    poses = poses[:imgs.shape[0]]
    # for spilt in spilts:
    out_data = {
        'fl_x': K[0][0],
        'fl_y': K[0][0],
        'cx': K[0][2],
        'cy': K[1][2],
        'w': imgs[0].shape[1],
        'h': imgs[0].shape[0],
        "bbx2w":listify_matrix(bbx),
        "inv_pose":listify_matrix(inv_pose),
        "scale":scale,
        "use_bbx": False,
        'aabb_scale': 16
        }
    out_data['frames'] = []

    # os.makedirs("kitti360", exist_ok=True)
    save_dir = f"kitti360_{config.start_index}_{config.num_images}"
    os.makedirs(save_dir,exist_ok=True)

    # mask_dir = save_dir +"/semantics/"
    # os.makedirs(mask_dir,exist_ok=True)
    

    for i in range(0,poses.shape[0]):
        frame_data = {
            'file_path': image_names[i]+".png",
            'transform_matrix': listify_matrix(poses[i])
        }
        filename =f'{save_dir}/'+ image_names[i]+".png"
        imageio.imwrite(filename,imgs[i])

        out_data['frames'].append(frame_data)

    with open(f'{save_dir}/transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

    



def _load_data(datadir,sequence ='2013_05_28_drive_0000_sync',start_index = -1, num = -1,use_fisheye = False):
    '''Load intrinstic matrix'''
    intrinstic_file = os.path.join(os.path.join(datadir, 'calibration'), 'perspective.txt')
    with open(intrinstic_file) as f:
        lines = f.readlines()
        for line in lines:
            lineData = line.strip().split()
            if lineData[0] == 'P_rect_00:':
                K_00 = np.array(lineData[1:]).reshape(3,4).astype(np.float)
                K_00 = K_00[:,:-1]
            elif lineData[0] == 'P_rect_01:':
                K_01 = np.array(lineData[1:]).reshape(3,4).astype(np.float)
                K_01 = K_01[:,:-1]
            elif lineData[0] == 'R_rect_01:':
                R_rect_01 = np.eye(4)
                R_rect_01[:3,:3] = np.array(lineData[1:]).reshape(3,3).astype(np.float)

    '''Load extrinstic matrix'''
    CamPose_00 = {}
    CamPose_01 = {}
    
    extrinstic_file = os.path.join(datadir,os.path.join('data_poses',sequence))
    cam2world_file_00 = os.path.join(extrinstic_file,'cam0_to_world.txt')
    pose_file = os.path.join(extrinstic_file,'poses.txt')


    ''' Camera_00  to world coordinate '''
    with open(cam2world_file_00,'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(map(float,line.strip().split()))
            CamPose_00[lineData[0]] = np.array(lineData[1:]).reshape(4,4)

    ''' Camera_01 to world coordiante '''
    CamToPose = loadCameraToPose(os.path.join(os.path.join(datadir, 'calibration'),'calib_cam_to_pose.txt'))
    CamToPose_01 = CamToPose['image_01:']
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0]
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    for frame, pose in zip(frames, poses):
        pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
        pp = np.matmul(pose, CamToPose_01)
        CamPose_01[frame] = np.matmul(pp, np.linalg.inv(R_rect_01))


 
    ''' Load corrlected images camera 00--> index    camera 01----> index+1'''
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)

    imgae_dir = os.path.join(datadir,sequence)
    image_00 = os.path.join(imgae_dir,'image_00/data_rect')
    image_01 = os.path.join(imgae_dir,'image_01/data_rect')


    all_images = []
    all_fish_images =[]
    all_poses = []
    all_img_name = []

    for idx in range(start_index,start_index+num,1):
        ## read image_00
        image = imread(os.path.join(image_00,"{:010d}.png").format(idx))
        all_images.append(image)
        all_poses.append(CamPose_00[idx])
        all_img_name.append(str(idx) + "_00")

        ## read image_01
        image = imread(os.path.join(image_01, "{:010d}.png").format(idx))
        all_images.append(image)
        all_poses.append(CamPose_01[idx])
        all_img_name.append(str(idx)  + "_01")
       
       
    imgs = np.stack(all_images,-1)
    imgs = np.moveaxis(imgs, -1, 0)
    c2w = np.stack(all_poses)

    
    return c2w,imgs, K_00, all_img_name

def load_fisheye(datadir,sequence ='2013_05_28_drive_0000_sync',start_index = -1, num = -1):
    '''Load intrinstic matrix'''
    intrinsic_fisheye_02_dir = os.path.join(os.path.join(datadir, 'calibration'),"image_02.yaml")
    intrinsic_fisheye_03_dir = os.path.join(os.path.join(datadir, 'calibration'),"image_03.yaml")
    intrinsic_fisheye_02 = readYAMLFile(intrinsic_fisheye_02_dir)
    intrinsic_fisheye_03 = readYAMLFile(intrinsic_fisheye_03_dir)

    '''Load extrinstic matrix'''
    extrinstic_file = os.path.join(datadir,os.path.join('data_poses',sequence))
    pose_file = os.path.join(extrinstic_file,'poses.txt')
    CamToPose = loadCameraToPose(os.path.join(os.path.join(datadir, 'calibration'),'calib_cam_to_pose.txt'))
    poses = np.loadtxt(pose_file)
    frames = poses[:,0]
    poses = np.reshape(poses[:,1:],[-1,3,4])

    ''' Camera_02 Camera_03 to world coordiante '''
    CamPose_02 = {}
    CamPose_03 = {}
    for frame, pose in zip(frames, poses): 
        pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
        CamPose_02[frame] = np.matmul(pose, CamToPose['image_02:'])
        CamPose_03[frame] = np.matmul(pose, CamToPose['image_03:'])
    
       
    ''' Load corrlected images camera 00--> index    camera 01----> index+1'''
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)
        
     ## read fisheye images
    imgae_dir = os.path.join(datadir,sequence)
    image_02 = os.path.join(imgae_dir,'image_02/data_rgb')
    image_03 = os.path.join(imgae_dir,'image_03/data_rgb')


    all_fish_images =[]
    all_poses = []
    all_img_name = []

    for idx in range(start_index,start_index+num,1):
        ## read image_02
        image = imread(os.path.join(image_02, "{:010d}.png").format(idx))
        all_fish_images.append(image)
        all_poses.append(CamPose_02[idx])
        all_img_name.append(str(idx)  + "_02")

        ## read image_03
        image = imread(os.path.join(image_03, "{:010d}.png").format(idx))
        all_fish_images.append(image)
        all_poses.append(CamPose_03[idx])
        all_img_name.append(str(idx)  + "_03")

    all_fish_images = np.stack(all_fish_images,-1)
    all_fish_images = np.moveaxis(all_fish_images, -1, 0)
    c2w = np.stack(all_poses)

    return c2w,all_fish_images, intrinsic_fisheye_02, intrinsic_fisheye_03, all_img_name

def readYAMLFile(fileName):
        '''make OpenCV YAML file compatible with python'''
        import re
        import yaml
        ret = {}
        skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
        with open(fileName,'r') as fin:
            for i in range(skip_lines):
                fin.readline()
            yamlFileOut = fin.read()
            myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
            yamlFileOut = myRe.sub(r': \1', yamlFileOut)
            ret = yaml.safe_load(yamlFileOut)
        return ret

def Normailize_T(poses,camera_type):
    cameara_type_matrix = np.array([1, -1, -1])
    if camera_type == "neus":
        cameara_type_matrix = np.array([1,1,1])


    # mid_frames = poses.shape[0]//2
    # mid_frames = 1
    # inv_pose = np.linalg.inv(poses[mid_frames])

    # import pdb; pdb.set_trace()


    avg_pose = average_poses(poses[:,:3])
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = avg_pose
    inv_pose = np.linalg.inv(pose_avg_homo)

    for i, pose in enumerate(poses):
        # if i == mid_frames:
        #     poses[i] = np.eye(4)
        # else:
        poses[i] = np.dot(inv_pose,poses[i])  ## 注意 inv_pose 乘是 左乘
        
    '''New Normalization '''
    # scale = np.max(poses[:,:3,3])
    scale = 1
    print(f"CG 系的坐标pose: scale:{scale}\n")
    for i in range(poses.shape[0]):
        poses[i,:3,3] = poses[i,:3,3]/scale
        poses[i,:3,:3] = poses[i,:3,:3] * cameara_type_matrix  ## opencv2openGL
        print(poses[i,:3,3])
    return poses, inv_pose, scale

def Normailize_T_fisheye(poses,camera_type,fisheye_poses):
    cameara_type_matrix = np.array([1, -1, -1,1])
    if camera_type == "neus":
        cameara_type_matrix = np.array([1,1,1,1])

    mid_frames = 0
    inv_pose = np.linalg.inv(poses[mid_frames])
    for i,pose in enumerate(poses):
        if i == mid_frames:
            poses[i] = np.eye(4) * cameara_type_matrix
        else:
            poses[i] = np.dot(inv_pose,poses[i]) *cameara_type_matrix
            
        fisheye_poses[i] = np.dot(inv_pose,fisheye_poses[i]) *cameara_type_matrix
    return np.concatenate([poses,fisheye_poses],axis=0)



def loadCameraToPose(filename):
    # open file
    Tr = {}
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(line.strip().split())
            if lineData[0] == 'image_01:':
                data = np.array(lineData[1:]).reshape(3,4).astype(np.float)
                data = np.concatenate((data,lastrow), axis=0)
                Tr[lineData[0]] = data
            elif lineData[0] == 'image_02:':
                    data = np.array(lineData[1:]).reshape(3, 4).astype(np.float)
                    data = np.concatenate((data, lastrow), axis=0)
                    Tr[lineData[0]] = data
            elif lineData[0] == 'image_03:':
                data = np.array(lineData[1:]).reshape(3, 4).astype(np.float)
                data = np.concatenate((data, lastrow), axis=0)
                Tr[lineData[0]] = data

    return Tr

if __name__ == '__main__':
    main('/work/Users/lisicheng/Dataset/KITTI-360/')
