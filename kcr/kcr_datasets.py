import functools
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import time
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable
import os
from mmdet.core import eval_recalls
from mmdet.datasets import CocoDataset

# from mmdet.datasets.builder import DATASETS
from mmrotate.datasets.builder import DATASETS
from mmdet.datasets.api_wrappers import COCO, COCOeval
from .utils.rotate import poly2obb_np_le90_seg, obb2poly_np_le90

# from mmrotate.core.bbox.transforms import obb2poly_np_le90
from mmrotate.datasets.dota import eval_rbbox_map as eval_map
import cv2
from mmdet.datasets.custom import CustomDataset
from torch.utils.data import Dataset
import random
from mmcv.parallel.data_container import DataContainer


@DATASETS.register_module()
class RotatedDataset(CocoDataset):
    # CLASSES = "banana"
    PALETTE = (220, 20, 60)

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        version="le90",
        test_class_idx=None,
    ):

        print("Loading with Rotated dataset.")
        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
        )
        self.test_class_idx = test_class_idx

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info["filename"].replace("jpg", "png")

        bboxes_angle = []
        bboxes_poly = []
        gt_labels_ = []
        for mask, label in zip(gt_masks_ann, gt_labels):
            # if len(mask) > 1:
                # print("Found multiple polygons")
            mask_new = []
            for m in mask:
                mask_new += m
            mask = mask_new
            mask = np.array(mask, dtype=np.float32)
            # else:
            #     mask = np.array(mask[0], dtype=np.float32)
            # # cv2.minAreaRect(mask.reshape(-1, 2))
            # try:
            #     convex_hull = cv2.convexHull(mask[:10].reshape(-1, 2))
            # except:
            #     print()
            bbox_angle = poly2obb_np_le90_seg(mask)
            if bbox_angle is None:
                continue
            bbox_poly = obb2poly_np_le90(bbox_angle)
            bboxes_angle.append(bbox_angle)
            bboxes_poly.append(bbox_poly)
            gt_labels_.append(label)

        if len(bboxes_angle) > 0:
            bboxes_angle = np.float32(np.stack(bboxes_angle))
            bboxes_poly = np.float32(np.stack(bboxes_poly))
            gt_labels = np.array(gt_labels_)
        else:
            bboxes_angle = np.ndarray((0, 5), dtype=np.float32)
            bboxes_poly = np.ndarray((0, 8), dtype=np.float32)
            gt_labels = np.ndarray((0,), dtype=np.float32)

        ann = dict(
            bboxes=bboxes_angle,
            polygons=bboxes_poly,
            labels=gt_labels,
            # bboxes_ignore=gt_bboxes_ignore.reshape(-1, 5),
            bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
            masks=gt_masks_ann,
            seg_map=seg_map,
            labels_ignore=np.array([]),
        )

        return ann

    def evaluate(
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,
        scale_ranges=None,
        nproc=4,
    ):
        """Evaluate the datasets.

        Args:
            results (list): Testing results of the datasets.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """

        if self.test_class_idx is not None:
            print(f"\nThe test class id is: {self.test_class_idx}")
            results = [
                result[self.test_class_idx:]
                for result in results
            ]

        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["mAP"]
        if metric not in allowed_metrics:
            raise KeyError(f"metric {metric} is not supported")
        annotations = [self.get_ann_info(i) for i in range(len(self))]

        # self.filter(annotations, results)

        eval_results = {}
        if metric == "mAP":
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                # version=self.version,
                logger=logger,
                nproc=nproc,
            )
            eval_results["mAP"] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def filter(self, annotations, results):
        for img_idx, (ann, result) in enumerate(zip(annotations, results)):
            area = ann["bboxes"][:, 2] * ann["bboxes"][:, 3]
            area_threshold = np.median(area) / 2
            valid_ann = area > area_threshold
            ann["bboxes"] = ann["bboxes"][valid_ann]
            ann["polygons"] = ann["polygons"][valid_ann]
            ann["labels"] = ann["labels"][valid_ann]
            del ann["masks"]

            area_result = result[0][:, 2] * result[0][:, 3]
            valid_result = area_result > area_threshold
            result[0] = result[0][valid_result]


@DATASETS.register_module()
class KCRAADataset(RotatedDataset):
    def __init__(self, enlarge_factor=1, **kwargs):
        self.enlarge_factor = enlarge_factor
        super().__init__(**kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info["filename"].replace("jpg", "png")

        bboxes_angle = []
        bboxes_poly = []
        # for mask in gt_masks_ann:
        #     if len(mask) > 1:
        #         print("Found multiple polygons")
        #     else:
        #         mask = np.array(mask[0], dtype=np.float32)
        #     # # cv2.minAreaRect(mask.reshape(-1, 2))
        #     # try:
        #     #     convex_hull = cv2.convexHull(mask[:10].reshape(-1, 2))
        #     # except:
        #     #     print()
        #     bbox_angle = poly2obb_np_le90_seg(mask)
        #     bbox_poly = obb2poly_np_le90(bbox_angle)
        #
        #     bboxes_angle.append(bbox_angle)
        #     bboxes_poly.append(bbox_poly)

        for box in gt_bboxes:
            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            x1_ = cx - (w / 2) * self.enlarge_factor
            x2_ = cx + (w / 2) * self.enlarge_factor
            y1_ = cy - (h / 2) * self.enlarge_factor
            y2_ = cy + (h / 2) * self.enlarge_factor

            # bbox_angle = np.array([cx, cy, w, h, 0])

            bbox_poly = np.array(
                [x1_, y1_, x2_, y1_, x2_, y2_, x1_, y2_], dtype=np.float32
            )
            bbox_angle = poly2obb_np_le90_seg(bbox_poly)
            bboxes_angle.append(bbox_angle)
            bboxes_poly.append(bbox_poly)

        bboxes_angle = np.float32(np.stack(bboxes_angle))
        bboxes_poly = np.float32(np.stack(bboxes_poly))

        ann = dict(
            bboxes=bboxes_angle,
            polygons=bboxes_poly,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore.reshape(-1, 5),
            masks=gt_masks_ann,
            seg_map=seg_map,
            labels_ignore=np.array([]),
        )

        return ann


@DATASETS.register_module()
class KCRDataset(Dataset):
    def __init__(self, rotated_cfg, aa_cfg):
        # Get classes on each dataset
        classes_rotated = rotated_cfg["classes"]
        classes_aa = aa_cfg["classes"]

        self.CLASSES = classes_rotated + classes_aa

        self.aa_start_index = len(classes_rotated)

        self.rotated_dataset = RotatedDataset(**rotated_cfg)
        self.aa_dataset = KCRAADataset(**aa_cfg)

        self.flag = self.aa_dataset.flag

    def __getitem__(self, aa_idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        # if self.test_mode:
        #     return self.aa_dataset.prepare_test_img(aa_idx)

        # Get AA Data
        while True:
            aa_data = self.aa_dataset.prepare_train_img(aa_idx)
            if aa_data is None:
                aa_idx = self.aa_dataset._rand_another(aa_idx)
                continue
            break

        # Get Rotated Data
        rotated_idx = random.randint(0, len(self.rotated_dataset) - 1)
        while True:
            rotated_data = self.rotated_dataset.prepare_train_img(rotated_idx)
            if rotated_data is None:
                rotated_idx = self.rotated_dataset._rand_another(rotated_idx)
                continue
            break

        aa_gt_labels = DataContainer(aa_data["gt_labels"].data + self.aa_start_index)
        aa_data["gt_labels"] = aa_gt_labels
        # data = aa_data
        return rotated_data, aa_data

    def __len__(self):
        return len(self.aa_dataset)


@DATASETS.register_module()
class KCRDatasetSC(Dataset):
    def __init__(self, rotated_cfg, aa_cfg):
        # Get classes on each dataset
        classes_rotated = rotated_cfg["classes"]
        classes_aa = aa_cfg["classes"]

        self.CLASSES = classes_aa

        self.aa_start_index = len(classes_rotated)

        self.rotated_dataset = RotatedDataset(**rotated_cfg)
        self.aa_dataset = KCRAADataset(**aa_cfg)
        # self.aa_dataset = JohariRotatedDataset(**aa_cfg)

        self.flag = self.aa_dataset.flag

    def __getitem__(self, aa_idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        # if self.test_mode:
        #     return self.aa_dataset.prepare_test_img(aa_idx)

        # Get AA Data
        while True:
            aa_data = self.aa_dataset.prepare_train_img(aa_idx)
            if aa_data is None:
                aa_idx = self.aa_dataset._rand_another(aa_idx)
                continue
            break

        # Get Rotated Data
        rotated_idx = random.randint(0, len(self.rotated_dataset) - 1)
        while True:
            rotated_data = self.rotated_dataset.prepare_train_img(rotated_idx)
            num_obj = len(rotated_data["gt_labels"].data)
            if rotated_data is None:
                rotated_idx = self.rotated_dataset._rand_another(rotated_idx)
                continue
            break

        rotated_data["gt_labels"] = DataContainer(
            torch.zeros_like(rotated_data["gt_labels"].data)
        )
        aa_data["gt_labels"] = DataContainer(
            torch.zeros_like(aa_data["gt_labels"].data)
        )
        # data = aa_data
        return rotated_data, aa_data

    def __len__(self):
        return len(self.aa_dataset)


@DATASETS.register_module()
class KCRDatasetMultiClass(KCRDataset):
    pass