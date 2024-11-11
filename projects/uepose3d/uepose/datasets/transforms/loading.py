# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import LoadImageFromFile
import mmengine.fileio as fileio
from mmpose.registry import TRANSFORMS
import mmcv

@TRANSFORMS.register_module()
class LoadStereoImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """
    def load_stereo_imgs(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        def load_image(filename):
            try:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e
            # in some cases, images are not read successfully, the img would be
            # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
            assert img is not None, f'failed to load image: {filename}'
            if self.to_float32:
                img = img.astype(np.float32)
            return img

        if 'img_path' in results and 'img_paths' not in results:
            left_img = load_image(results['img_path'])
            right_img = load_image(results['img_path'])
        elif 'img_path' not in results and 'img_paths' in results:
        # filename = results['img_path']
            left_img = load_image(results['img_paths'][0][0])
            right_img = load_image(results['img_paths'][0][1])
        results['img'] = left_img
        results['right_img'] = right_img
        
        
        results['img_shape'] = left_img.shape[:2]
        results['ori_shape'] = left_img.shape[:2]
        return results
    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = self.load_stereo_imgs(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results
