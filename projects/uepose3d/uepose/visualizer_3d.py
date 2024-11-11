import math
from typing import Dict, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
import cv2
import mmcv
import numpy as np
from mmpose.registry import VISUALIZERS
from mmpose.apis import convert_keypoint_definition
from mmpose.visualization import Pose3dLocalVisualizerPlus
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData

@VISUALIZERS.register_module()
class StereoPose3dLocalVisualizerPlus(Pose3dLocalVisualizerPlus):
    
    def __init__(
            self,
            name: str = 'visualizer',
            image: Optional[np.ndarray] = None,
            vis_backends: Optional[Dict] = None,
            save_dir: Optional[str] = None,
            bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
            kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
            link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
            text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
            skeleton: Optional[Union[List, Tuple]] = None,
            line_width: Union[int, float] = 1,
            radius: Union[int, float] = 3,
            show_keypoint_weight: bool = False,
            backend: str = 'opencv',
            alpha: float = 0.8,
            det_kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
            det_dataset_skeleton: Optional[Union[str,
                                                 Tuple[Tuple[int]]]] = None,
            det_dataset_link_color: Optional[np.ndarray] = None):
        
        super().__init__(name, image, vis_backends, save_dir, bbox_color,
                         kpt_color, link_color, text_color, skeleton,
                         line_width, radius, show_keypoint_weight, backend,
                         alpha,det_kpt_color,det_dataset_skeleton,det_dataset_link_color)
        
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       det_data_sample: Optional[PoseDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_2d: bool = True,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       dataset_2d: str = 'coco',
                       dataset_3d: str = 'h36m',
                       convert_keypoint: bool = True,
                       axis_azimuth: float = 70,
                       axis_limit: float = 1,
                       axis_dist: float = 10.0,
                       axis_elev: float = 15.0,
                       num_instances: int = -1,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0,
                       show_right_view: bool= False,
                       **kwargs) -> None:
        det_img_data = None
        scores_2d = None

        if draw_2d:
            det_img_data = image.copy()

            # draw bboxes & keypoints
            if (det_data_sample is not None
                    and 'pred_instances' in det_data_sample):
                det_img_data, scores_2d = self._draw_instances_kpts(
                    image=det_img_data,
                    instances=det_data_sample.pred_instances,
                    kpt_thr=kpt_thr,
                    show_kpt_idx=show_kpt_idx,
                    skeleton_style=skeleton_style)
                if draw_bbox:
                    det_img_data = self._draw_instances_bbox(
                        det_img_data, det_data_sample.pred_instances)
        if scores_2d is not None and convert_keypoint:
            if scores_2d.ndim == 2:
                scores_2d = scores_2d[..., None]
            scores_2d = np.squeeze(
                convert_keypoint_definition(scores_2d, dataset_2d, dataset_3d),
                axis=-1)
        pred_img_data = self._draw_3d_data_samples(
            image.copy(),
            data_sample,
            draw_gt=draw_gt,
            num_instances=num_instances,
            axis_azimuth=axis_azimuth,
            axis_limit=axis_limit,
            show_kpt_idx=show_kpt_idx,
            axis_dist=axis_dist,
            axis_elev=axis_elev,
            scores_2d=scores_2d)

        # merge visualization results
        if det_img_data is not None:
            width = max(pred_img_data.shape[1] - det_img_data.shape[1], 0)
            height = max(pred_img_data.shape[0] - det_img_data.shape[0], 0)
            det_img_data = cv2.copyMakeBorder(
                det_img_data,
                height // 2,
                (height // 2 + 1) if height % 2 == 1 else height // 2,
                width // 2, (width // 2 + 1) if width % 2 == 1 else width // 2,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255))
            drawn_img = np.concatenate((det_img_data, pred_img_data), axis=1)
        else:
            drawn_img = pred_img_data

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # save drawn_img to backends
            self.add_image(name, drawn_img, step)

        return self.get_image()
    
    def _draw_3d_data_samples(self,
                              image: np.ndarray,
                              pose_samples: PoseDataSample,
                              draw_gt: bool = True,
                              kpt_thr: float = 0.3,
                              num_instances=-1,
                              axis_azimuth: float = 70,
                              axis_limit: float = 1.7,
                              axis_dist: float = 10.0,
                              axis_elev: float = 15.0,
                              show_kpt_idx: bool = False,
                              scores_2d: Optional[np.ndarray] = None,
                              **kwargs):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            num_instances (int): Number of instances to be shown in 3D. If
                smaller than 0, all the instances in the pose_result will be
                shown. Otherwise, pad or truncate the pose_result to a length
                of num_instances.
            axis_azimuth (float): axis azimuth angle for 3D visualizations.
            axis_dist (float): axis distance for 3D visualizations.
            axis_elev (float): axis elevation view angle for 3D visualizations.
            axis_limit (float): The axis limit to visualize 3d pose. The xyz
                range will be set as:
                - x: [x_c - axis_limit/2, x_c + axis_limit/2]
                - y: [y_c - axis_limit/2, y_c + axis_limit/2]
                - z: [0, axis_limit]
                Where x_c, y_c is the mean value of x and y coordinates
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            scores_2d (np.ndarray, optional): Keypoint scores of 2d estimation
                that will be used to filter 3d instances.

        Returns:
            Tuple(np.ndarray): the drawn image which channel is RGB.
        """
        vis_width = max(image.shape)
        vis_height = vis_width

        if 'pred_instances' in pose_samples:
            pred_instances = pose_samples.pred_instances
        else:
            pred_instances = InstanceData()
        if num_instances < 0:
            if 'keypoints' in pred_instances:
                num_instances = len(pred_instances)
            else:
                num_instances = 0
        else:
            if len(pred_instances) > num_instances:
                pred_instances_ = InstanceData()
                for k in pred_instances.keys():
                    new_val = pred_instances[k][:num_instances]
                    pred_instances_.set_field(new_val, k)
                pred_instances = pred_instances_
            elif num_instances < len(pred_instances):
                num_instances = len(pred_instances)

        num_fig = num_instances
        if draw_gt:
            vis_width *= 2
            num_fig *= 2

        plt.ioff()
        fig = plt.figure(
            figsize=(vis_width * num_instances * 0.01, vis_height * 0.01))

        def _draw_3d_instances_kpts(keypoints,
                                    scores,
                                    scores_2d,
                                    keypoints_visible,
                                    fig_idx,
                                    show_kpt_idx,
                                    title=None):
            
            for idx, (kpts, score, score_2d) in enumerate(
                    zip(keypoints, scores, scores_2d)):

                valid = np.logical_and(score >= kpt_thr, score_2d >= kpt_thr,
                                       np.any(~np.isnan(kpts), axis=-1))

                kpts_valid = kpts[valid]
                ax = fig.add_subplot(
                    1, num_fig, fig_idx * (idx+1), projection='3d')
                ax.view_init(elev=axis_elev, azim=axis_azimuth)
                ax.set_aspect('auto')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                if title:
                    ax.set_title(f'{title} ({idx})')
                ax.dist = axis_dist

                x_c = np.mean(kpts_valid[:, 0]) if valid.any() else 0
                y_c = np.mean(kpts_valid[:, 1]) if valid.any() else 0
                z_c = np.mean(kpts_valid[:, 2]) if valid.any() else 0

                ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
                ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])
                ax.set_zlim3d(
                    [min(0, z_c - axis_limit / 2), z_c + axis_limit / 2])

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                x_3d, y_3d, z_3d = np.split(kpts_valid[:, :3], [1, 2], axis=1)

                kpt_color = kpt_color[valid] / 255.

                ax.scatter(x_3d, y_3d, z_3d, marker='o', c=kpt_color)

                if show_kpt_idx:
                    for kpt_idx in range(len(x_3d)):
                        ax.text(x_3d[kpt_idx][0], y_3d[kpt_idx][0],
                                z_3d[kpt_idx][0], str(kpt_idx))

                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        sk_indices = [_i for _i in sk]
                        if np.sum(kpts[sk_indices[0]])==0.0 or  np.sum(kpts[sk_indices[1]])==0.0:
                            continue
                        xs_3d = kpts[sk_indices, 0]
                        ys_3d = kpts[sk_indices, 1]
                        zs_3d = kpts[sk_indices, 2]
                        kpt_score = score[sk_indices]
                        kpt_score_2d = score_2d[sk_indices]
                        if kpt_score.min() > kpt_thr and kpt_score_2d.min(
                        ) > kpt_thr:
                            # matplotlib uses RGB color in [0, 1] value range
                            _color = link_color[sk_id] / 255.
                            ax.plot(
                                xs_3d, ys_3d, zs_3d, color=_color, zdir='z')

        if 'keypoints' in pred_instances:
            keypoints = pred_instances.get('keypoints',
                                           pred_instances.keypoints)

            if 'keypoint_scores' in pred_instances:
                scores = pred_instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if scores_2d is None:
                scores_2d = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in pred_instances:
                keypoints_visible = pred_instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            _draw_3d_instances_kpts(keypoints, scores, scores_2d,
                                    keypoints_visible, 1, show_kpt_idx,
                                    'Prediction')

        if draw_gt and 'gt_instances' in pose_samples:
            gt_instances = pose_samples.gt_instances
            if 'lifting_target' in gt_instances:
                keypoints = gt_instances.get('lifting_target',
                                             gt_instances.lifting_target)
                scores = np.ones(keypoints.shape[:-1])

                if 'lifting_target_visible' in gt_instances:
                    keypoints_visible = gt_instances.lifting_target_visible
                else:
                    keypoints_visible = np.ones(keypoints.shape[:-1])
            elif 'keypoints_gt' in gt_instances:
                keypoints = gt_instances.get('keypoints_gt',
                                             gt_instances.keypoints_gt)
                scores = np.ones(keypoints.shape[:-1])

                if 'keypoints_visible' in gt_instances:
                    keypoints_visible = gt_instances.keypoints_visible
                else:
                    keypoints_visible = np.ones(keypoints.shape[:-1])
            else:
                raise ValueError('to visualize ground truth results, '
                                 'data sample must contain '
                                 '"lifting_target" or "keypoints_gt"')

            if scores_2d is None:
                scores_2d = np.ones(keypoints.shape[:-1])

            _draw_3d_instances_kpts(keypoints, scores, scores_2d,
                                    keypoints_visible, 2, show_kpt_idx,
                                    'Ground Truth')

        # convert figure to numpy array
        fig.tight_layout()
        fig.canvas.draw()

        pred_img_data = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)

        if not pred_img_data.any():
            pred_img_data = np.full((vis_height, vis_width, 3), 255)
        else:
            width, height = fig.get_size_inches() * fig.get_dpi()
            pred_img_data = pred_img_data.reshape(
                int(height),
                -1, 3)

        plt.close(fig)

        return pred_img_data