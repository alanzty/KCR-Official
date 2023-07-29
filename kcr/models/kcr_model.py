from mmrotate.models.builder import ROTATED_DETECTORS, ROTATED_HEADS
from mmrotate.models.detectors.two_stage import RotatedTwoStageDetector
from mmrotate.models.dense_heads.oriented_rpn_head import OrientedRPNHead
from mmrotate.models.roi_heads.oriented_standard_roi_head import OrientedStandardRoIHead
from mmcv.runner import force_fp32
import torch
from mmdet.core import images_to_levels, multi_apply
from mmrotate.core.bbox.transforms import obb2hbb_le90, obb2xyxy_le90, obb2hbb_oc
from mmrotate.core import rbbox2roi, build_assigner
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import (
    RotatedShared2FCBBoxHead,
)
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox import MaxIoUAssigner
from mmdet.core.bbox.assigners.assign_result import AssignResult
from math import pi


@ROTATED_DETECTORS.register_module()
class KCROrientedRCNN(RotatedTwoStageDetector):
    def __init__(
        self,
        backbone,
        rpn_head,
        roi_head,
        train_cfg,
        test_cfg,
        neck=None,
        pretrained=None,
        init_cfg=None,
        src_asp_comp=None
    ):
        super(KCROrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.src_asp_comp = src_asp_comp

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs
    ):

        bs = int(len(img_metas) / 2)

        rotated_img = img[:bs]
        rotated_img_metas = img_metas[:bs]
        rotated_gt_bboxes = gt_bboxes[:bs]
        rotated_gt_labels = gt_labels[:bs]

        aa_img = img[bs:]
        aa_img_metas = img_metas[bs:]
        aa_gt_bboxes = gt_bboxes[bs:]
        aa_gt_labels = gt_labels[bs:]

        losses_rotated = self.forward_train_rotated(
            rotated_img, rotated_img_metas, rotated_gt_bboxes, rotated_gt_labels
        )
        losses_aa = self.forward_train_aa(
            aa_img, aa_img_metas, aa_gt_bboxes, aa_gt_labels
        )

        losses = dict()

        for key in losses_aa:
            # if "cls" in key or 'rpn' in key:
            #     losses["aa_" + key] = losses_aa[key]
            losses["aa_" + key] = losses_aa[key]
            losses["rotated_" + key] = losses_rotated[key]
        delete_keys = set()
        for key, val in losses.items():
            if type(val) is list:
                for v in val:
                    if not torch.isfinite(v):
                        delete_keys.add(key)
            elif not torch.isfinite(val):
                delete_keys.add(key)
        for key in delete_keys:
            print("Found NAN!", key)
            del losses[key]
        return losses

    def forward_train_rotated(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs
    ):

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs
        )
        losses.update(roi_losses)

        return losses

    def forward_train_aa(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs
    ):

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                gt_style="aa",
                **kwargs
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            gt_style="aa",
            **kwargs
        )
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        outputs = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

        if self.src_asp_comp:
            for output in outputs[0]:
                ratio = output[:, 2] / output[:, 3]
                output[ratio < self.src_asp_comp[0], 3] /= (1.1 + self.src_asp_comp[1]/10)
        return outputs


@ROTATED_HEADS.register_module()
class KCROrientedRPNHead(OrientedRPNHead):
    def forward_train(
        self,
        x,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        gt_style="rotated",
        **kwargs
    ):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_style=gt_style
        )
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg
            )
            return losses, proposal_list

    @force_fp32(apply_to=("cls_scores", "bbox_preds"))
    def loss(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        gt_style="rotated",
    ):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device
        )
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos
        )

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            gt_style=gt_style,
        )
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def loss_single(
        self,
        cls_score,
        bbox_pred,
        anchors,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        num_total_samples,
        gt_style="rotated",
    ):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples
        )
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 6)
        bbox_weights = bbox_weights.reshape(-1, 6)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 6)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        if gt_style == "aa":
            loss_bbox = self.loss_bbox(
                bbox_pred[..., :4],
                bbox_targets[..., :4],
                bbox_weights[..., :4],
                avg_factor=num_total_samples,
            )
        else:
            loss_bbox = self.loss_bbox(
                bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples
            )
        return loss_cls, loss_bbox


@ROTATED_HEADS.register_module()
class KCROrientedStandardRoIHead(OrientedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""

    def __init__(self, **kwargs):
        super(KCROrientedStandardRoIHead, self).__init__(**kwargs)

        if self.train_cfg:
            self.aa_bbox_assigner = build_assigner(self.train_cfg.aa_assigner)

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        gt_style="rotated",
    ):
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                if gt_style == "aa":
                    proposal_hbb = torch.cat(
                        (
                            obb2hbb_le90(proposal_list[i][:, :5]),
                            proposal_list[i][:, -1:],
                        ),
                        dim=-1,
                    )
                    assign_result = self.bbox_assigner.assign(
                        proposal_hbb, gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                    )
                    # assign_result = self.aa_bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i], img_metas[i])

                else:
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_bboxes_ignore[i],
                        gt_labels[i],
                    )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = (
                        gt_bboxes[i].new((0, gt_bboxes[0].size(-1))).zero_()
                    )
                else:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i][
                        sampling_result.pos_assigned_gt_inds, :
                    ]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas, gt_style=gt_style
            )
            losses.update(bbox_results["loss_bbox"])

        return losses

    def _bbox_forward_train(
        self, x, sampling_results, gt_bboxes, gt_labels, img_metas, gt_style="rotated"
    ):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg, gt_style=gt_style
        )

        # if gt_style == "aa":
        #     labels, label_weights, box_targets, bbox_weights = bbox_targets
        #
        #     bbox_targets = labels, label_weights, box_targets, bbox_weights

        loss_bbox = self.bbox_head.loss(
            bbox_results["cls_score"], bbox_results["bbox_pred"], rois, *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results


@ROTATED_HEADS.register_module()
class KCRBBoxHead(RotatedShared2FCBBoxHead):
    def get_targets(
        self,
        sampling_results,
        gt_bboxes,
        gt_labels,
        rcnn_train_cfg,
        concat=True,
        gt_style="rotated",
    ):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg,
            gt_style=gt_style,
        )

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def _get_target_single(
        self,
        pos_bboxes,
        neg_bboxes,
        pos_gt_bboxes,
        pos_gt_labels,
        cfg,
        gt_style="rotated",
    ):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets

            if gt_style == "aa":
                long_side, _ = pos_gt_bboxes[:, 2:4].max(-1)
                short_side, _ = pos_gt_bboxes[:, 2:4].min(-1)
                aspect_ratio = long_side / short_side
                # bbox_weights[:num_pos, :2] = 1
                # bbox_weights[:num_pos][aspect_ratio > 3] = 0.5
                # bbox_weights[:num_pos, 2:] = torch.clip(0.1 * aspect_ratio.unsqueeze(-1), 0, 1)

            else:
                bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights


@BBOX_ASSIGNERS.register_module()
class KCRaaAssigner(MaxIoUAssigner):
    # def __init__(self, asp_thr=3, gt_area_thr=0.01, **kwargs):
    #     super(KCRaaAssigner, self).__init__(**kwargs)
    #     self.asp_thr = asp_thr
    #     self.gt_area_thr = gt_area_thr
    #
    # def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, img_meta=None):
    #     assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
    #             gt_bboxes.shape[0] > self.gpu_assign_thr) else False
    #     # compute overlap and assign gt on CPU when number of GT is large
    #     if assign_on_cpu:
    #         device = bboxes.device
    #         bboxes = bboxes.cpu()
    #         gt_bboxes = gt_bboxes.cpu()
    #         if gt_bboxes_ignore is not None:
    #             gt_bboxes_ignore = gt_bboxes_ignore.cpu()
    #         if gt_labels is not None:
    #             gt_labels = gt_labels.cpu()
    #
    #     long_side, _ = gt_bboxes[:, 2:4].max(-1)
    #     short_side, _ = gt_bboxes[:, 2:4].min(-1)
    #     aspect_ratio = long_side / short_side
    #
    #     overlaps = self.iou_calculator(gt_bboxes, bboxes)
    #     iof = self.iou_calculator(bboxes, gt_bboxes, mode="iof")
    #     iof = iof.transpose(0, 1)
    #
    #     gt_area = long_side * short_side / (img_meta['pad_shape'][0] * img_meta['pad_shape'][1])
    #
    #     if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
    #             and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
    #         if self.ignore_wrt_candidates:
    #             ignore_overlaps = self.iou_calculator(
    #                 bboxes, gt_bboxes_ignore, mode='iof')
    #             ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
    #         else:
    #             ignore_overlaps = self.iou_calculator(
    #                 gt_bboxes_ignore, bboxes, mode='iof')
    #             ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
    #         overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
    #
    #     assign_result = self.assign_wrt_overlaps(overlaps, gt_labels, aspect_ratio=aspect_ratio, bboxes=bboxes, iof=iof, gt_area=gt_area)
    #     if assign_on_cpu:
    #         assign_result.gt_inds = assign_result.gt_inds.to(device)
    #         assign_result.max_overlaps = assign_result.max_overlaps.to(device)
    #         if assign_result.labels is not None:
    #             assign_result.labels = assign_result.labels.to(device)
    #     return assign_result
    #
    # def assign_wrt_overlaps(self, overlaps, gt_labels=None, aspect_ratio=None, bboxes=None, iof=None, gt_area=None):
    #     num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
    #
    #     # 1. assign -1 by default
    #     assigned_gt_inds = overlaps.new_full((num_bboxes, ),
    #                                          -1,
    #                                          dtype=torch.long)
    #
    #     if num_gts == 0 or num_bboxes == 0:
    #         # No ground truth or boxes, return empty assignment
    #         max_overlaps = overlaps.new_zeros((num_bboxes, ))
    #         if num_gts == 0:
    #             # No truth, assign everything to background
    #             assigned_gt_inds[:] = 0
    #         if gt_labels is None:
    #             assigned_labels = None
    #         else:
    #             assigned_labels = overlaps.new_full((num_bboxes, ),
    #                                                 -1,
    #                                                 dtype=torch.long)
    #         return AssignResult(
    #             num_gts,
    #             assigned_gt_inds,
    #             max_overlaps,
    #             labels=assigned_labels)
    #
    #     # for each anchor, which gt best overlaps with it
    #     # for each anchor, the max iou of all gts
    #     max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    #     # for each gt, which anchor best overlaps with it
    #     # for each gt, the max iou of all proposals
    #     gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
    #
    #     # KCR: Determine threshold by aspect ratio
    #     # neg_iou_thr = torch.ones_like(max_overlaps) * self.neg_iou_thr
    #     # pos_iou_thr = torch.ones_like(max_overlaps) * self.pos_iou_thr
    #     aspect_ratio_matched = aspect_ratio[argmax_overlaps]
    #     gt_area_matched = gt_area[argmax_overlaps]
    #     angle_predicted = pi/4 - abs(abs(bboxes[:, 4]) - pi/4)
    #     # rotational_matches = torch.logical_and(aspect_ratio_matched < 2, angle_predicted > pi/8)
    #     # neg_iou_thr[rotational_matches] = 0.5
    #     # pos_iou_thr[rotational_matches] = 0.5
    #
    #     # iof_thr = torch.ones_like(pos_iou_thr) * 1.1
    #     # iou_thr = torch.ones_like(pos_iou_thr) * 0.5
    #     # max_iof, argmax_iof = iof.max(dim=0)
    #
    #     # 2. assign negative: below
    #     # the negative inds are set to be 0
    #     # if isinstance(self.neg_iou_thr, float):
    #     hc = (aspect_ratio_matched >= self.asp_thr) | (gt_area_matched < self.gt_area_thr)
    #     lc = (aspect_ratio_matched < self.asp_thr) & (gt_area_matched > self.gt_area_thr)
    #
    #     neg_indices1 = lc & (max_overlaps < 0.1)
    #     neg_indices2 = hc & (max_overlaps < 0.5)
    #     neg_indices = neg_indices1 | neg_indices2
    #     assigned_gt_inds[(max_overlaps >= 0) & neg_indices] = 0
    #         # assigned_gt_inds[torch.logical_and((max_overlaps < iou_thr), max_iof < iof_thr)] = 0
    #     # elif isinstance(self.neg_iou_thr, tuple):
    #     #     assert len(self.neg_iou_thr) == 2
    #     #     assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0
    #
    #     # 3. assign positive: above positive IoU threshold
    #     # pos_inds1 = (max_overlaps >= pos_iou_thr) & (aspect_ratio_matched >= 3)
    #     # pos_inds2 = (aspect_ratio_matched < 3) & (max_overlaps > 0.5)
    #     # pos_inds = pos_inds1 & pos_inds2
    #     # pos_inds = (max_overlaps > 0.5)
    #     pos_inds1 = (max_overlaps > 0.5) & hc
    #     pos_inds2 = (max_overlaps > 0.5) & lc & (angle_predicted > pi/8)
    #     pos_inds = pos_inds1 | pos_inds2
    #     # pos_inds = torch.logical_or((max_overlaps >= iou_thr), max_iof >= iof_thr)
    #     assigned_gt_inds[pos_inds1] = argmax_overlaps[pos_inds1] + 1
    #
    #     if self.match_low_quality:
    #         # Low-quality matching will overwrite the assigned_gt_inds assigned
    #         # in Step 3. Thus, the assigned gt might not be the best one for
    #         # prediction.
    #         # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
    #         # bbox 1 will be assigned as the best target for bbox A in step 3.
    #         # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
    #         # assigned_gt_inds will be overwritten to be bbox 2.
    #         # This might be the reason that it is not used in ROI Heads.
    #         for i in range(num_gts):
    #             if gt_max_overlaps[i] >= self.min_pos_iou:
    #                 if self.gt_max_assign_all:
    #                     max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
    #                     assigned_gt_inds[max_iou_inds] = i + 1
    #                 else:
    #                     assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
    #
    #     if gt_labels is not None:
    #         assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
    #         pos_inds = torch.nonzero(
    #             assigned_gt_inds > 0, as_tuple=False).squeeze()
    #         if pos_inds.numel() > 0:
    #             assigned_labels[pos_inds] = gt_labels[
    #                 assigned_gt_inds[pos_inds] - 1]
    #     else:
    #         assigned_labels = None
    #
    #     return AssignResult(
    #         num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    pass
