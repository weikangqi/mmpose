import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import sys
import os.path as osp
import json
from rtmlib import Body, draw_skeleton
sys.path.append(os.path.join(os.path.dirname(__file__),'metric_depth'))
from depth_anything_v2.dpt import DepthAnythingV2
from util.metric import eval_depth

def calculate_oks(gt, pred,vis, sigma=0.1):
    minx,maxx,miny,maxy = np.min(gt[:,0]),np.max(gt[:,0]),np.min(gt[:,1]),np.max(gt[:,1])
    w,h = maxx-minx, maxy-miny
    # 计算OKS
    distance = np.sqrt(np.sum((gt - pred) ** 2, axis=-1))
    oks = np.exp(-distance**2 / (w*h*2 * sigma**2)) * (vis>0)
    oks = np.sum(oks) / np.sum(vis>0)
    return oks

def calculate_oks_matrix(gt_keypoints, pred_keypoints,vis):
    M, N = len(gt_keypoints), len(pred_keypoints)
    oks_matrix = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            oks_matrix[i, j] = calculate_oks(gt_keypoints[i], pred_keypoints[j],vis[i])
    
    return oks_matrix

def calculate_best_matches(gt_keypoints, pred_keypoints,vis, threshold=0.5):
    oks_matrix = calculate_oks_matrix(gt_keypoints, pred_keypoints,vis)
    max_oks_per_gt = np.max(oks_matrix, axis=1)
    best_matches = np.argmax(oks_matrix, axis=1)
    
    # 过滤掉低于阈值的匹配
    best_matches[max_oks_per_gt < threshold] = -1
    
    return best_matches, max_oks_per_gt


def eval_mpjpe(depth_model,data_dir,tag='test'):
    annotation_path = osp.join(data_dir,f'annotations/{tag}.json')
    images_dir = osp.join(data_dir,f'{tag}')
    openpose_skeleton = False 

    anno_json = json.load(open(annotation_path,'r'))
    images_anno = anno_json['images']
    poses_anno = anno_json['annotations']
    
    device = 'cpu'  # cpu, cuda, mps
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    body_model = Body(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)
    
    
    sum_mpjpe = []
    for image_anno in images_anno:
        id = image_anno['id']
        
        image_path = osp.join(images_dir,image_anno['file_name'])
        raw_image = cv2.imread(image_path)
        gt_kp2d = []
        gt_kp3d = []
        for pose_anno in poses_anno:
            if pose_anno['image_id'] == id:
                gt_kp2d.append(np.array(pose_anno['keypoints']).reshape(-1,3))
                gt_kp3d.append(np.array(pose_anno['keypoints_3d']).reshape(-1,3))
        depth = depth_model.infer_image(raw_image, 518)
        gt_kp2d = np.array(gt_kp2d)
        gt_kp3d = np.array(gt_kp3d)
        pose2d_index = np.floor(gt_kp2d[:,:,0:2]).astype(np.int32)
        K = np.array(
            [[320,0,320],
            [0,320,240],
            [0,0,1]]
        )
        
        pred_keypoints, scores = body_model(raw_image)
        pred_keypoints = np.array(pred_keypoints)
        
        pred_keypoints[pred_keypoints[:,:,0]>=640,0] = 640 - 1
        pred_keypoints[pred_keypoints[:,:,1]>=480,1] = 480 - 1
        pred_keypoints[pred_keypoints[:,:,0]<0,0] = 0
        pred_keypoints[pred_keypoints[:,:,1]<0,1] = 0
        
        best_match,_ = calculate_best_matches(gt_kp2d[:,:17,0:2],pred_keypoints,gt_kp2d[:,:17,2],0.5)
        for i in range(pred_keypoints.shape[0]):
            pid = best_match[i]
            pred_p_kp2d = np.zeros([17,3])
            pred_p_kp2d[:,:2]= pred_keypoints[i]
            # pred_p_kp2d = pred_p_kp2d[:,np.newaxis]
            pred_p_kp2d[:,2] = 1
            # gt_p_kp2d[:,2] = 1
            gt_p_kp3d = gt_kp3d[pid][:17,:]
            pose2d_idx =  np.floor(pred_p_kp2d[:,0:2]).astype(np.int32)
            z = depth[pose2d_idx[:,1],pose2d_idx[:,0]].reshape(17,1)
            tmp =  z * pred_p_kp2d 
            tmp = tmp.reshape(-1,3)
            pred_p_kp3d = []
            for kid in range(17):
                t = - np.linalg.inv(K) @ tmp[kid]
                pred_p_kp3d.append(t)
            pred_p_kp3d = np.array(pred_p_kp3d)
            flag = (gt_p_kp3d[:,1]!=0)
            flag = flag.reshape(-1,1)
            pred_p_kp3d = pred_p_kp3d * flag
            # pred_kp3d.append(pred_p_kp3d)
            dis = np.linalg.norm(gt_p_kp3d - pred_p_kp3d,axis=1)
            mpjpe = np.mean(dis[dis!=0]) * 1000
            # print(mpjpe)
            sum_mpjpe.append(mpjpe)
            
            
            # gt_matched = 
            
        
        # for pid in range(pose2d_index.shape[0]):
        #     gt_p_kp2d = gt_kp2d[pid]
        #     gt_p_kp2d[:,2] = 1
        #     gt_p_kp3d = gt_kp3d[pid]
        #     pose2d_idx = pose2d_index[pid]
        #     z = depth[pose2d_idx[:,1],pose2d_idx[:,0]].reshape(21,1)
        #     tmp =  z * gt_p_kp2d 
        #     tmp = tmp.reshape(-1,3)
        #     pred_p_kp3d = []
        #     for kid in range(21):
        #         t = - np.linalg.inv(K) @ tmp[kid]
        #         pred_p_kp3d.append(t)
        #     pred_p_kp3d = np.array(pred_p_kp3d)
        #     flag = (gt_p_kp3d[:,1]!=0)
        #     flag = flag.reshape(-1,1)
        #     pred_p_kp3d = pred_p_kp3d * flag
        #     # pred_kp3d.append(pred_p_kp3d)
        #     dis = np.linalg.norm(gt_p_kp3d - pred_p_kp3d,axis=1)
        #     mpjpe = np.mean(dis[dis!=0]) * 1000
        #     # print(mpjpe)
        #     sum_mpjpe.append(mpjpe)
    print(np.average(np.array(mpjpe)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    # parser.add_argument('--img-path', type=str, default='assets/examples')
    parser.add_argument('--img-path', type=str, default='/workspace/mmpose3d/data/uecoco/test')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/workspace/mmpose3d/work_dirs/mono_pose')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--checkpoint', type=str,default='/workspace/mmpose3d/data/latest.pth')
    parser.add_argument('--max-depth', type=float,  default=10)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    DEVICE = 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    #depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    # print(torch.load(f'/mnt/tbdisk/wkq/code/Depth-Anything-V2/metric_depth/exp/UE4/latest.pth', map_location='cpu')["model"])
    params_dict = torch.load(args.checkpoint, map_location='cpu')["model"]
    new_state_dict = {}
    for key, value in params_dict.items():
    # 如果键以 "module." 开头，去掉 "module."
        if key.startswith("module."):
            new_key = key.replace("module.", "", 1)  # 只替换第一个 "module."
        else:
            new_key = key  # 如果键不以 "module." 开头，保持不变
        
        # 将新键和值添加到新的字典中
        new_state_dict[new_key] = value
    # depth_anything.load_state_dict(torch.load(f'/mnt/tbdisk/wkq/code/Depth-Anything-V2/metric_depth/exp/UE4/latest.pth', map_location='cpu'))
    depth_anything.load_state_dict(new_state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    
    eval_mpjpe(depth_anything,'/workspace/MobileHumanPose3D/dataset/uecoco_3d')
    
    # if os.path.isfile(args.img_path):
    #     if args.img_path.endswith('txt'):
    #         with open(args.img_path, 'r') as f:
    #             filenames = f.read().splitlines()
    #     else:
    #         filenames = [args.img_path]
    # else:
    #     filenames = sorted(glob.glob(os.path.join(args.img_path, '*.png'), recursive=True))
    # # depth_files = sorted(glob.glob(os.path.join(args.img_path, '*.png'), recursive=True))
    
    # os.makedirs(args.outdir, exist_ok=True)
    
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    # error_sum = []
    # # 将图片和depth 路径写入txt
    # # 图片和depth 每一对写一行 绝对路径
    # # 绝对路径

    # for k, filename in enumerate(filenames):
    #     print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
    #     raw_image = cv2.imread(filename)
    #     # gt_depth =  np.load(depth_gt )/100.
    #     depth = depth_anything.infer_image(raw_image, args.input_size)
    #     # error_sum.append(eval_depth(depth,gt_depth)['abs_rel'])
    #     depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    #     depth = depth.astype(np.uint8)
      
    #     if args.grayscale:
    #         depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    #     else:
    #         depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
    #     if args.pred_only:
    #         cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
    #     else:
    #         split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    #         combined_result = cv2.hconcat([raw_image, split_region, depth])
            
    #         cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
# print(f'Average error: {np.mean(error_sum)}')