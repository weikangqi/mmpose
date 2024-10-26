import sys
import os
# sys.path.append('core_rt')
sys.path.append(os.path.join(os.path.dirname(__file__),'core_rt'))

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import time
import logging
import numpy as np
import torch
import os.path as osp
import json
from tqdm import tqdm
from rtmlib import Body, draw_skeleton
import cv2
from rt_igev_stereo import IGEVStereo, autocast
from utils.utils import InputPadder
import stereo_datasets as datasets
from utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


@torch.no_grad()
def validate_pose3d(depth_model,data_dir,tag='test', iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    # model.eval()
    annotation_path = osp.join(data_dir,f'annotations/{tag}.json')
    images_dir = osp.join(data_dir,f'{tag}')
    openpose_skeleton = False 

    anno_json = json.load(open(annotation_path,'r'))
    images_anno = anno_json['images']
    poses_anno = anno_json['annotations']
    
    device = 'cpu'  # cpu, cuda, mps
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    body_model = Body(to_openpose=openpose_skeleton,pose='rtmo',
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)
    
    
    sum_mpjpe = []
    DEVICE = 'cuda'
    for image_anno in images_anno:
        id = image_anno['id']
        
        left_image_path = osp.join(images_dir,image_anno['left_file_name'])
        right_image_path = osp.join(images_dir,image_anno['right_file_name'])
        left_raw_image = cv2.imread(left_image_path)
        right_raw_image = cv2.imread(right_image_path)
        gt_kp2d = []
        gt_kp3d = []
        for pose_anno in poses_anno:
            if pose_anno['image_id'] == id:
                gt_kp2d.append(np.array(pose_anno['keypoints']).reshape(-1,3))
                gt_kp3d.append(np.array(pose_anno['keypoints_3d']).reshape(-1,3))
        # depth = depth_model.infer_image(raw_image, 518)
        def load_image(img):
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to(DEVICE)
        image1 = load_image(left_raw_image)
        image2 = load_image(right_raw_image)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        disp = depth_model(image1, image2, iters=16, test_mode=True)
        disp = padder.unpad(disp).cpu()
        disp = torch.permute(disp,[2,3,0,1])[:,:,0,0].numpy()
        
        
        depth = 320 * 0.02 / (disp+1e-8)
        
        gt_kp2d = np.array(gt_kp2d)
        gt_kp3d = np.array(gt_kp3d)
        pose2d_index = np.floor(gt_kp2d[:,:,0:2]).astype(np.int32)
        K = np.array(
            [[320,0,320],
            [0,320,240],
            [0,0,1]]
        )
        
        pred_keypoints, scores = body_model(left_raw_image)
        pred_keypoints = np.array(pred_keypoints)
        
        pred_keypoints[pred_keypoints[:,:,0]>=640,0] = 640 - 1
        pred_keypoints[pred_keypoints[:,:,1]>=480,1] = 480 - 1
        pred_keypoints[pred_keypoints[:,:,0]<0,0] = 0
        pred_keypoints[pred_keypoints[:,:,1]<0,1] = 0
        
        best_match,_ = calculate_best_matches(gt_kp2d[:,:17,0:2],pred_keypoints,gt_kp2d[:,:17,2],0.5)
        for i in range(len(best_match)):
            pid = best_match[i]
            pred_p_kp2d = np.zeros([17,3])
            pred_p_kp2d[:,:2]= pred_keypoints[pid]
            # pred_p_kp2d = pred_p_kp2d[:,np.newaxis]
            pred_p_kp2d[:,2] = 1
            # gt_p_kp2d[:,2] = 1
            gt_p_kp3d = gt_kp3d[i][:17,:]
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
            print(mpjpe)
            sum_mpjpe.append(mpjpe)                                                                          
    
    
    # val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    # val_loader = data.DataLoader(val_dataset, batch_size=8, 
    #     pin_memory=True, shuffle=False, num_workers=8)

    # out_list, epe_list = [], []
    # for i_batch, (_, *data_blob) in enumerate(tqdm(val_loader)):
    #     image1, image2, disp_gt, valid_gt = [x for x in data_blob]

    #     image1 = image1.cuda()
    #     image2 = image2.cuda()

    #     padder = InputPadder(image1.shape, divis_by=32)
    #     image1, image2 = padder.pad(image1, image2)

    #     with autocast(enabled=mixed_prec):
    #         disp_pr = model(image1, image2, iters=iters, test_mode=True)
    #     disp_pr = padder.unpad(disp_pr).cpu()
    #     assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
    #     epe = torch.abs(disp_pr - disp_gt)

    #     epe = epe.flatten()
    #     val = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)
    #     if(np.isnan(epe[val].mean().item())):
    #         continue

    #     out = (epe > 3.0)
    #     epe_list.append(epe[val].mean().item())
    #     out_list.append(out[val].cpu().numpy())

    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)

    # epe = np.mean(epe_list)
    # d1 = 100 * np.mean(out_list)

    # f = open('test_sceneflow.txt', 'a')
    # f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    # print("Validation Scene Flow: %f, %f" % (epe, d1))
    # return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_rt/sceneflow.pth')
    # parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    # # Architecure choices
   # Architecure choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
    args = parser.parse_args()



    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    restore_ckpt = '/workspace/mmpose3d/data/sceneflow.pth'
    # if args.restore_ckpt is not None:
    # assert args.restore_ckpt.endswith(".pth")
    logging.info("Loading checkpoint...")
    checkpoint = torch.load(restore_ckpt)
    model.load_state_dict(checkpoint, strict=True)
    logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    data_dir = '/workspace/MobileHumanPose3D/dataset/uecoco_3d'
    validate_pose3d(model,data_dir,tag='test_stereo')

    # if args.dataset == 'eth3d':
    #     validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    # elif args.dataset == 'kitti':
    #     validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    # elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
    #     validate_middlebury(model, iters=args.valid_iters, resolution=args.dataset[-1], mixed_prec=args.mixed_precision)

    # elif args.dataset == 'sceneflow':
    #     validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)
